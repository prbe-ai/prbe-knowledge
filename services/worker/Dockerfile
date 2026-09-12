FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev git curl \
 && rm -rf /var/lib/apt/lists/*

# gitleaks — the credential scanner `engine/ingest/secret_redaction.py` shells
# to before any chunk is persisted. Pinned: the rule corpus is the detector, so
# an unpinned upgrade silently changes what we redact (and what we miss).
#
# WITHOUT THIS BINARY the redactor fails open — text passes through unredacted
# with a warning — so a missing install does not break ingestion. It just
# removes the backstop. /health reports `secret_scanner: false` so the absence
# is visible, but nothing REFUSES traffic over it (that would trade a silent
# gap for an outage), which is why the install is pinned here rather than left
# to a probe to catch.
ARG GITLEAKS_VERSION=8.30.1
RUN curl -fsSL -o /tmp/gitleaks.tar.gz \
      "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" \
 && tar -xzf /tmp/gitleaks.tar.gz -C /usr/local/bin gitleaks \
 && rm /tmp/gitleaks.tar.gz \
 && chmod 0755 /usr/local/bin/gitleaks \
 && gitleaks version

WORKDIR /app

COPY pyproject.toml ./
COPY engine/ ./engine/
COPY kb/ ./kb/
COPY services/ ./services/
COPY db/ ./db/
# scripts/ ships into the worker image so operators can run one-shot
# admin/maintenance scripts via `flyctl ssh console`. The deploy workflow's
# paths-filter includes 'scripts/**' under worker/poller so adding a new
# script triggers a worker rebuild that bakes it in.
COPY scripts/ ./scripts/
COPY alembic.ini ./

RUN pip install -e "."

# Worker is a long-running drain loop, not an HTTP server.
CMD ["python", "-m", "services.ingestion.worker"]
