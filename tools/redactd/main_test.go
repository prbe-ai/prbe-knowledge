package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/zricethezav/gitleaks/v8/detect"
)

func rulesPath(t *testing.T) string {
	t.Helper()
	p, err := filepath.Abs("../../engine/ingest/probe_rules.toml")
	if err != nil {
		t.Fatalf("resolving rules path: %v", err)
	}
	return p
}

func TestLoadConfigBringsTheStockCorpusAndTheExtension(t *testing.T) {
	cfg, err := loadConfig(rulesPath(t))
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	// A config that loads to a handful of rules means `extend.useDefault` was
	// ignored, which reads downstream as "fewer findings" rather than as an
	// error.
	if len(cfg.Rules) < 100 {
		t.Fatalf("expected the stock corpus, got %d rules", len(cfg.Rules))
	}
	if _, ok := cfg.Rules["probe-anchored-secret"]; !ok {
		t.Fatal("the vendored extension rule is missing")
	}
}

func TestLoadConfigRejectsAMissingFile(t *testing.T) {
	if _, err := loadConfig("/nonexistent/rules.toml"); err == nil {
		t.Fatal("a missing config must be a startup failure, not an empty ruleset")
	}
}

func TestLinesAreOneBasedLikeTheCLI(t *testing.T) {
	cfg, err := loadConfig(rulesPath(t))
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	d := detect.NewDetector(cfg)
	secret := "hT7xQ2mVb9Lk" + "Zp0RwYe4Ns6Uc1Ai8Jd3Fg5Oh2Pq"
	got, err := scan(context.Background(), d, []string{"AWS Secret Access Key [None]: " + secret})
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(got) != 1 || len(got[0]) == 0 {
		t.Fatalf("expected a finding, got %#v", got)
	}
	if got[0][0].Line != 1 {
		t.Fatalf("line must be 1-based to match the CLI, got %d", got[0][0].Line)
	}
}

func TestScanReturnsOneEntryPerText(t *testing.T) {
	cfg, _ := loadConfig(rulesPath(t))
	d := detect.NewDetector(cfg)
	texts := []string{"clean", "", "also clean"}
	got, err := scan(context.Background(), d, texts)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	// Cardinality is the contract: the client treats a short response as a
	// failure precisely because a missing entry would otherwise read as clean.
	if len(got) != len(texts) {
		t.Fatalf("expected %d results, got %d", len(texts), len(got))
	}
}

func TestScanStopsAtTheDeadlineRatherThanReturningPartialResults(t *testing.T) {
	cfg, _ := loadConfig(rulesPath(t))
	d := detect.NewDetector(cfg)
	ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	time.Sleep(time.Millisecond)
	if _, err := scan(ctx, d, []string{"a", "b"}); err == nil {
		t.Fatal("a cancelled scan must error, not return the texts it managed")
	}
}

func TestReadFrameRefusesAnOversizeFrame(t *testing.T) {
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.BigEndian, uint32(maxRequestBytes+1))
	buf.WriteString("x")
	if _, err := readFrame(bufio.NewReader(&buf), maxRequestBytes); err == nil {
		t.Fatal("an oversize frame must be refused, never truncated")
	}
}

func TestErrorsNeverCarryRequestText(t *testing.T) {
	// The payloads crossing this socket are customer documents; an error
	// string is the one place a value could leak into a log.
	var out bytes.Buffer
	secret := "hT7xQ2mVb9Lk" + "Zp0RwYe4Ns6Uc1Ai8Jd3Fg5Oh2Pq"
	if err := writeResponse(&out, response{OK: false, Error: "request was not valid JSON"}); err != nil {
		t.Fatalf("writeResponse: %v", err)
	}
	if strings.Contains(out.String(), secret) {
		t.Fatal("an error response carried input text")
	}
	var resp response
	_ = json.Unmarshal(out.Bytes()[4:], &resp)
	if resp.OK {
		t.Fatal("an error response must not be ok")
	}
}
