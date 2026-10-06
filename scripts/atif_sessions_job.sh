#!/usr/bin/env bash
# Run scripts/atif_sessions.py as a throwaway Job built from the live engine
# worker's pod spec: the same image, env and secrets (DB DSN, R2 credentials,
# redactd on PATH), its own pod, its own limits.
#
# Why not `kubectl exec ... python` into the worker: an exec'd process shares
# the serving container's memory limit, and the OOM killer picks the biggest
# process there, which is the server (research retrieval, 2026-09-29).
#
#   scripts/atif_sessions_job.sh replay --all-tenants --sample 500
#   scripts/atif_sessions_job.sh backfill --all-tenants            # dry run
#   scripts/atif_sessions_job.sh backfill --all-tenants --write
#
# The script must be in the image the worker runs (merge, then the data-plane
# image build, then the worker restart). Output is ids, counts and timings only.
set -euo pipefail

ctx="${KUBE_CONTEXT:-do-sfo3-probe-research}"
ns="${NAMESPACE:-research}"
deploy="${SOURCE_DEPLOYMENT:-research-os-engine-worker}"
mode="${1:?usage: $0 <replay|backfill> [args...]}"
name="atif-sessions-${mode}-$(date -u +%Y%m%d%H%M%S)"
cmd=$(python3 -c 'import json, sys; print(json.dumps(["python", "-m", "scripts.atif_sessions", *sys.argv[1:]]))' "$@")

kubectl --context "$ctx" -n "$ns" get deploy "$deploy" -o json | jq \
  --arg name "$name" --argjson cmd "$cmd" '
  {
    apiVersion: "batch/v1",
    kind: "Job",
    metadata: {name: $name, namespace: .metadata.namespace,
               labels: {"app.kubernetes.io/name": "atif-sessions"}},
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
               | {image, env, envFrom, volumeMounts, securityContext}
               + {name: "atif-sessions", command: $cmd,
                  resources: {requests: {cpu: "250m", memory: "1Gi"},
                              limits: {cpu: "1", memory: "3Gi"}}}]})
      }
    }
  } | del(..|nulls)' | kubectl --context "$ctx" -n "$ns" create -f -

echo "follow: kubectl --context $ctx -n $ns logs -f job/$name"
