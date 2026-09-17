// redactd: the credential scanner, booted once instead of once per document.
//
// WHY THIS EXISTS
//
// The ingestion worker scans every document it persists by running the
// gitleaks CLI as a subprocess. Measured in the pod on 2026-09-16: one spawn
// costs ~1.05 CPU-seconds, of which 0.67 is `gitleaks version` -- the Go
// runtime and the embedded 222-rule corpus starting up, before a byte of input
// is read. Input size, rule set and GOMAXPROCS do not move it. At 15,617
// documents a day that is 4.6 CPU-hours of pure process startup, and it is
// what pinned the worker at 81.5% CFS throttling and pushed the queue's median
// wait to 33 minutes.
//
// gitleaks has no server mode, but it is a Go library. This is that library
// behind a Unix socket: the rules compile once at boot and a scan costs
// milliseconds. Same engine, same rules file, so there is no compatibility
// port to get wrong -- and tests/test_redactd_differential.py proves the
// library and the CLI agree finding-for-finding on the outage corpus.
//
// WHAT IT DELIBERATELY DOES NOT DO
//
//   - It never logs, returns, or errors with a secret VALUE. Findings carry a
//     rule id, the matched value (which the caller needs to replace it) and a
//     line number, and every error string is constructed from the request
//     SHAPE only. The value travels over a 0600 socket on the pod's own
//     filesystem and nowhere else.
//   - It has no network listener. A TCP port would make a credential scanner
//     reachable from anywhere in the cluster.
//   - It does not decide anything. "Could not scan" is an error the caller
//     turns into a retry; this process never says "clean" on its own behalf.
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/spf13/viper"
	"github.com/zricethezav/gitleaks/v8/config"
	"github.com/zricethezav/gitleaks/v8/detect"
)

const (
	// Matches engine/ingest/secret_redaction.py. A request over this is
	// rejected rather than truncated: truncation is how the tail of a large
	// transcript went unscanned for months.
	maxRequestBytes = 8 << 20
	// A response is findings only, never input. This bounds a pathological
	// document producing thousands of hits.
	maxResponseBytes = 1 << 20
	maxTexts         = 512
	// Per-request ceiling. The client's own timeout is longer, so a scan that
	// hits this returns a clean error rather than the client giving up on a
	// still-running server.
	scanDeadline = 30 * time.Second
	// Concurrent scans. The worker runs six claim loops; more in flight than
	// that only adds contention on a pod whose CPU limit is 2.
	maxInFlight = 4
)

type request struct {
	Op    string   `json:"op"`
	Texts []string `json:"texts"`
}

type finding struct {
	Rule   string `json:"rule"`
	Secret string `json:"secret"`
	Line   int    `json:"line"`
}

type response struct {
	OK       bool        `json:"ok"`
	Error    string      `json:"error,omitempty"`
	Findings [][]finding `json:"findings,omitempty"`
}

// readFrame reads one length-prefixed frame. Length prefixing rather than
// newline delimiting because the payload is arbitrary document text.
func readFrame(r *bufio.Reader, max int) ([]byte, error) {
	var n uint32
	if err := binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	if int(n) > max {
		return nil, fmt.Errorf("frame of %d bytes exceeds the %d byte ceiling", n, max)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return buf, nil
}

func writeFrame(w io.Writer, payload []byte) error {
	if err := binary.Write(w, binary.BigEndian, uint32(len(payload))); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

// writeResponse never lets an error string carry caller text: `err` here is
// always constructed from shapes and counts.
func writeResponse(w io.Writer, resp response) error {
	payload, err := json.Marshal(resp)
	if err != nil {
		payload, _ = json.Marshal(response{OK: false, Error: "response could not be encoded"})
	}
	if len(payload) > maxResponseBytes {
		payload, _ = json.Marshal(response{
			OK:    false,
			Error: fmt.Sprintf("response of %d bytes exceeds the %d byte ceiling", len(payload), maxResponseBytes),
		})
	}
	return writeFrame(w, payload)
}

func loadConfig(path string) (config.Config, error) {
	v := viper.New()
	v.SetConfigFile(path)
	if err := v.ReadInConfig(); err != nil {
		return config.Config{}, fmt.Errorf("reading %s: %w", path, err)
	}
	var vc config.ViperConfig
	if err := v.Unmarshal(&vc); err != nil {
		return config.Config{}, fmt.Errorf("parsing %s: %w", path, err)
	}
	cfg, err := vc.Translate()
	if err != nil {
		return config.Config{}, fmt.Errorf("translating %s: %w", path, err)
	}
	if len(cfg.Rules) == 0 {
		// A rules file that loads to zero rules is indistinguishable from a
		// clean corpus at every later point. Refuse to start.
		return config.Config{}, errors.New("config produced zero rules")
	}
	return cfg, nil
}

func scan(ctx context.Context, d *detect.Detector, texts []string) ([][]finding, error) {
	out := make([][]finding, len(texts))
	for i, text := range texts {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("scan deadline reached after %d of %d texts", i, len(texts))
		}
		hits := d.DetectString(text)
		found := make([]finding, 0, len(hits))
		for _, h := range hits {
			if h.Secret == "" || h.RuleID == "" {
				continue
			}
			// +1: DetectString counts lines from 0, while the CLI's stdin
			// path wraps input in a source that counts from 1. The number
			// reaches a customer-visible record, so the two transports must
			// not disagree about it -- proven by the differential test, which
			// compares (rule, secret, LINE).
			found = append(found, finding{Rule: h.RuleID, Secret: h.Secret, Line: h.StartLine + 1})
		}
		out[i] = found
	}
	return out, nil
}

func handle(conn net.Conn, d *detect.Detector, slots chan struct{}) {
	defer conn.Close()
	r := bufio.NewReaderSize(conn, 64<<10)
	for {
		raw, err := readFrame(r, maxRequestBytes)
		if err != nil {
			if !errors.Is(err, io.EOF) {
				_ = writeResponse(conn, response{OK: false, Error: err.Error()})
			}
			return
		}
		var req request
		if err := json.Unmarshal(raw, &req); err != nil {
			// Never echo the body: on a partial write it can hold the secret.
			_ = writeResponse(conn, response{OK: false, Error: "request was not valid JSON"})
			continue
		}
		switch req.Op {
		case "ping":
			// The readiness handshake. Socket existence is not readiness: a
			// stale socket file from a crashed process accepts nothing, and a
			// booting one has not compiled its rules yet.
			if err := writeResponse(conn, response{OK: true}); err != nil {
				return
			}
			continue
		case "scan":
		default:
			_ = writeResponse(conn, response{OK: false, Error: "unknown op"})
			continue
		}
		if len(req.Texts) > maxTexts {
			_ = writeResponse(conn, response{
				OK:    false,
				Error: fmt.Sprintf("%d texts exceeds the %d ceiling", len(req.Texts), maxTexts),
			})
			continue
		}

		slots <- struct{}{}
		ctx, cancel := context.WithTimeout(context.Background(), scanDeadline)
		findings, err := scan(ctx, d, req.Texts)
		cancel()
		<-slots

		if err != nil {
			_ = writeResponse(conn, response{OK: false, Error: err.Error()})
			continue
		}
		if err := writeResponse(conn, response{OK: true, Findings: findings}); err != nil {
			return
		}
	}
}

func main() {
	socketPath := flag.String("socket", "", "unix socket to listen on")
	configPath := flag.String("config", "", "gitleaks rules file (absolute)")
	check := flag.Bool("check", false, "load the rules, report, and exit")
	flag.Parse()

	// Findings are the only thing worth saying, and they are said over the
	// socket. Anything this process writes to stderr is an operational event,
	// never content.
	log.SetFlags(0)
	log.SetPrefix("redactd: ")

	if *configPath == "" || !filepath.IsAbs(*configPath) {
		log.Fatal("--config must be an absolute path")
	}
	cfg, err := loadConfig(*configPath)
	if err != nil {
		log.Fatalf("%v", err)
	}
	if *check {
		if _, ok := cfg.Rules["probe-anchored-secret"]; !ok {
			// The extension rule is the one this deployment adds on top of the
			// stock corpus. Loading without it means the vendored config was
			// ignored, which reads downstream as "no findings".
			log.Fatal("rules loaded but probe-anchored-secret is missing")
		}
		fmt.Printf("ok rules=%d\n", len(cfg.Rules))
		return
	}
	if *socketPath == "" {
		log.Fatal("--socket is required")
	}

	detector := detect.NewDetector(cfg)

	// A socket file left by a crashed predecessor accepts no connections and
	// blocks the bind.
	if err := os.Remove(*socketPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		log.Fatalf("clearing stale socket: %v", err)
	}
	ln, err := net.Listen("unix", *socketPath)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	// 0600: the payloads crossing this socket are the customer documents this
	// process exists to look for credentials in.
	if err := os.Chmod(*socketPath, 0o600); err != nil {
		log.Fatalf("chmod socket: %v", err)
	}

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-sigs
		_ = ln.Close()
	}()

	log.Printf("ready rules=%d socket=%s", len(cfg.Rules), *socketPath)
	slots := make(chan struct{}, maxInFlight)
	for {
		conn, err := ln.Accept()
		if err != nil {
			return // listener closed: shutdown
		}
		go handle(conn, detector, slots)
	}
}
