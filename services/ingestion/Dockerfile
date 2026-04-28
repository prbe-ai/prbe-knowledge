FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY shared/ ./shared/
COPY services/ ./services/
COPY db/ ./db/
COPY alembic.ini ./

RUN pip install -e "."

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
