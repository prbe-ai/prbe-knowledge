# Security Policy

## Reporting a vulnerability

If you believe you have found a security vulnerability in this project,
please report it privately. **Do not open a public issue, pull request, or
discussion for security reports.**

Email **security@prbe.ai** with:

- a description of the issue and its impact,
- steps to reproduce (a proof of concept, if you have one), and
- any affected versions or configuration.

Please give us a reasonable window to investigate and ship a fix before any
public disclosure. We aim to acknowledge a report within **3 business days**
and to provide a remediation timeline within **10 business days**. We follow a
**90-day** coordinated-disclosure window by default and will keep you updated
as we work through a fix.

## Supported versions

Security fixes are applied to the latest released version on the default
branch. Older versions are not maintained; please upgrade to the latest
release before reporting.

## Scope

This repository is the knowledge engine. Reports about a hosted deployment you
do not operate should go to the operator of that deployment. Configuration
secrets (API keys, tokens, encryption keys) are supplied at runtime via
environment variables and are never committed to this repository — see
`.env.example`.
