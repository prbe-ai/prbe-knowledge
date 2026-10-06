#!/usr/bin/env bash
# Run scripts/atif_sessions.py as a throwaway Job built from the live engine
# worker's pod spec: its env and secrets (DB DSN, R2 credentials, redactd on
# PATH), its own pod, its own limits -- and the image DIGEST a running worker pod
# runs. On research the Deployment names `:latest` with pullPolicy Always, so the
# tag alone would pull whatever was built last, not the code that serves.
#
# Why not `kubectl exec ... python` into the worker: an exec'd process shares
# the serving container's memory limit, and the OOM killer picks the biggest
# process there, which is the server (research retrieval, 2026-09-29).
#
#   scripts/atif_sessions_job.sh replay --all-tenants --sample 500
#   scripts/atif_sessions_job.sh backfill --all-tenants            # dry run
#   scripts/atif_sessions_job.sh backfill --all-tenants --write
#   DRY_RUN=1 scripts/atif_sessions_job.sh replay ...   # server-side validation only
#
# The script must be in the image the worker runs (merge, then the data-plane
# image build, then the worker restart). Output is ids, counts and timings only;
# the Job's `atif-sessions/image` annotation names the build it ran.
set -euo pipefail

ctx="${KUBE_CONTEXT:-do-sfo3-probe-research}"
ns="${NAMESPACE:-research}"
deploy="${SOURCE_DEPLOYMENT:-research-os-engine-worker}"
mode="${1:?usage: $0 <replay|backfill|strip> [args...]}"
# The script it runs: scripts.atif_sessions (replay, backfill) by default,
# MODULE=scripts.strip_session_payloads for `strip`.
module="${MODULE:-scripts.atif_sessions}"
name="atif-sessions-${mode}-$(date -u +%Y%m%d%H%M%S)"
cmd=$(python3 -c 'import json, sys; print(json.dumps(["python", "-m", sys.argv[1], *sys.argv[2:]]))' "$module" "$@")
selector=$(kubectl --context "$ctx" -n "$ns" get deploy "$deploy" -o json \
  | jq -r '.spec.selector.matchLabels | to_entries | map("\(.key)=\(.value)") | join(",")')
container=$(kubectl --context "$ctx" -n "$ns" get deploy "$deploy" -o jsonpath='{.spec.template.spec.containers[0].name}')
image=$(kubectl --context "$ctx" -n "$ns" get pods -l "$selector" --field-selector=status.phase=Running -o json \
  | jq -r --arg c "$container" '[.items[].status.containerStatuses[] | select(.name == $c) | .imageID] | unique | if length == 1 then .[0] else error("worker pods run \(length) different images: \(.)") end' \
  | sed 's#^docker-pullable://##')
case "$image" in *@sha256:*) ;; *) echo "no image digest from $deploy pods: '$image'" >&2; exit 1 ;; esac
echo "image: $image"

kubectl --context "$ctx" -n "$ns" get deploy "$deploy" -o json | jq \
  --arg name "$name" --argjson cmd "$cmd" --arg image "$image" '
  {
    apiVersion: "batch/v1",
    kind: "Job",
    metadata: {name: $name, namespace: .metadata.namespace,
               labels: {"app.kubernetes.io/name": "atif-sessions"},
               annotations: {"atif-sessions/image": $image}},
    spec: {
      ttlSecondsAfterFinished: 86400,
      backoffLimit: 0,
      activeDeadlineSeconds: 21600,
      template: {
        metadata: {labels: {"app.kubernetes.io/name": "atif-sessions"}},
        spec: (.spec.template.spec
          | {imagePullSecrets, volumes, securityContext}
          + {restartPolicy: "Never",
             containers: [.containers[0]
               | {env, envFrom, volumeMounts, securityContext}
               + {name: "atif-sessions", image: $image, imagePullPolicy: "IfNotPresent",
                  command: $cmd,
                  # Request what it may use: the node is packed on requests.
                  resources: {requests: {cpu: "250m", memory: "2Gi"},
                              limits: {cpu: "1", memory: "2Gi"}}}]})
      }
    }
  } | del(..|nulls)' | kubectl --context "$ctx" -n "$ns" create ${DRY_RUN:+--dry-run=server -o yaml} -f -

[ -n "${DRY_RUN:-}" ] || echo "follow: kubectl --context $ctx -n $ns logs -f job/$name"
