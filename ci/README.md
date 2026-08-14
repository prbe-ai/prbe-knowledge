# CI cluster access

`agent-optimization-nightly.yml` and `nightly-improvement-resume.yml` reach the
`probe-managed` DOKS cluster with a Kubernetes ServiceAccount, not with
`doctl`.

## Why this is not `doctl kubernetes cluster kubeconfig save`

That command mints a **fresh DigitalOcean personal access token on every
invocation** — `doks:probe-managed`, 7-day expiry, and nothing ever deletes it.
Six workflows across three repos called it, leaving 200+ live credentials on
the DO account, the great majority showing *Last Used: Never*.

That was not merely untidy. On **2026-08-12** the real
`DIGITALOCEAN_ACCESS_TOKEN` in `prbe-data-plane-image` hit its 90-day expiry,
`fleet-rollout` began failing, and the managed fleet stopped receiving every
engine build for 26 hours. One meaningful token expiring inside 20 pages of
machine-generated ones is not something anybody spots.

**This repo was next.** Its `DIGITALOCEAN_ACCESS_TOKEN` was minted 2026-05-18
and would have expired **2026-08-16**, taking both nightlies down the same way.
The credential was removed rather than renewed.

`doctl` was used here for nothing but fetching a kubeconfig.

## What exists

`rbac/probe-managed.yaml` creates, on the `probe-managed` cluster:

| object | purpose |
| --- | --- |
| namespace `ci` | shared home for CI identities; also used by prbe-data-plane-image |
| ServiceAccount `ci/knowledge-nightly` | this repo's identity |
| Secret `ci/knowledge-nightly-token` | its non-expiring token |
| RoleBinding → `admin` in `managed` | read `managed-retrieval`'s image tag; apply / wait / log / delete the trace-digest Job |

**Scope is `managed` only** — strictly narrower than the `ci/fleet-rollout`
ServiceAccount used by `prbe-data-plane-image`, which additionally needs
`control-plane` and `prbe-image-prepull`. Two identities rather than one shared
credential so that revoking the nightlies cannot affect fleet rollouts, and so
cluster audit logs attribute an action to the workflow that took it.

Verified denied: `control-plane`, `prbe-image-prepull`, `list namespaces`,
`kube-system` secrets.

Namespace-admin rather than a hand-enumerated verb list because these workflows
`kubectl apply` a rendered Job from `k8s/jobs/nightly-trace-digest.yaml` — a
Role pinned to today's fields breaks the next time that manifest gains one.

## Why the token does not expire

The Secret is of type `kubernetes.io/service-account-token`, which issues a
**non-expiring** token. The cluster is v1.35.1, so this Secret is what makes a
token exist at all — automatic creation alongside the ServiceAccount stopped in
v1.24.

`kubectl create token --duration` was rejected deliberately: it needs a working
credential in hand to mint the next one, which reintroduces exactly the
bootstrap dependency this change deletes, and adds a refresh job whose silent
failure would look identical to the outage above.

**The trade, stated plainly.** This swaps an expiring credential for a standing
one. An expired token fails loudly and stops everything; a leaked long-lived
token is quiet and lasts forever. The scoped RBAC is one mitigation; the
rotation plan below is the rest.

## Rotation plan

Manual and not calendar-driven — there is no expiry to force it. Rotate when
someone with repo-secret access leaves, when a run log is suspected of leaking
the token, when the RBAC is widened, or annually.

```sh
kubectl --context do-sfo3-probe-managed -n ci delete secret knowledge-nightly-token
kubectl --context do-sfo3-probe-managed apply -f ci/rbac/probe-managed.yaml

kubectl --context do-sfo3-probe-managed -n ci get secret knowledge-nightly-token \
  -o jsonpath='{.data.token}' | base64 -d |
  gh secret set KUBE_SA_TOKEN --repo prbe-ai/prbe-knowledge
```

Deleting the old Secret invalidates the old token immediately.

## First-time setup / disaster recovery

```sh
kubectl --context do-sfo3-probe-managed apply -f ci/rbac/probe-managed.yaml

R=prbe-ai/prbe-knowledge
C=do-sfo3-probe-managed
kubectl --context $C config view --minify \
  -o jsonpath='{.clusters[0].cluster.server}' | gh secret set KUBE_API_SERVER --repo $R
kubectl --context $C -n ci get secret knowledge-nightly-token \
  -o jsonpath='{.data.ca\.crt}' | gh secret set KUBE_CA_CERT --repo $R
kubectl --context $C -n ci get secret knowledge-nightly-token \
  -o jsonpath='{.data.token}' | base64 -d | gh secret set KUBE_SA_TOKEN --repo $R
```

`KUBE_CA_CERT` is stored **base64-encoded**, exactly as it appears in the
Secret — that is the form `certificate-authority-data` wants, so do not decode
it on the way in.

## Revoking this repo's access

```sh
kubectl --context do-sfo3-probe-managed -n ci delete sa knowledge-nightly
kubectl --context do-sfo3-probe-managed -n ci delete secret knowledge-nightly-token
kubectl --context do-sfo3-probe-managed -n managed delete rolebinding knowledge-nightly-admin
```

This leaves `prbe-data-plane-image`'s fleet-rollout access intact.
