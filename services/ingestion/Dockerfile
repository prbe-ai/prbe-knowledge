# Match the module's pinned toolchain floor; compile the same scanner for every
# standalone ingress/worker image. No Go runtime or compiler ships in the final image.
FROM golang:1.24.11-bookworm AS redactd-build
WORKDIR /build/redactd
COPY tools/redactd/go.mod tools/redactd/go.sum ./
RUN go mod download
COPY tools/redactd/ ./
RUN CGO_ENABLED=0 go build -mod=readonly -trimpath -ldflags="-s -w" -o /out/redactd .

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
 && rm -rf /var/lib/apt/lists/*

# Admission and normalization fail closed when scanner inspection is unavailable.
# Install the default daemon and the explicit CLI fallback with the same pinned
# gitleaks rule corpus. The rules file is included below with engine/.
COPY --from=redactd-build /out/redactd /usr/local/bin/redactd
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
COPY alembic.ini ./

RUN pip install -e "."

# scripts/ holds the generic DB migrate entrypoint (scripts/migrate.py) used by
# the compose / Helm one-shot migrate step. Copied after the install so it
# never busts the dependency-layer cache.
COPY scripts/ ./scripts/

EXPOSE 8080

# Run alembic upgrade head first, then start uvicorn.
# `release_command` in fly.toml runs migrations separately; keeping this
# fallback here so `fly deploy --local-only` also migrates.
#
# `::` binds the IPv6 wildcard, which on Linux is dual-stack and accepts
# both IPv4 and IPv6. Required for app-to-app calls via Fly's 6PN
# private network (`<app>.internal` DNS), which is IPv6-only — bare
# `0.0.0.0` would silently fail on .internal traffic.
CMD alembic upgrade head && \
    uvicorn services.ingestion.main:app --host :: --port 8080 --workers 2
