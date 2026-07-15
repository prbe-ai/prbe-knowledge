FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libigraph-dev: C library required by python-igraph + leidenalg
#   (used by the leiden community-detection cron).
# build-essential: compiles native extensions.
# libpq-dev: asyncpg.
# curl: ops debugging on the cron machine.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev curl libigraph-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Generic cron image — must include every module any scheduled command
# might import. Today: services.synthesis (wiki nightly trigger) and
# services.community + scripts.leiden_one_shot (leiden cron). Future
# cron commands land here too. Trade a slightly bigger image for one
# fly app instead of N.
COPY pyproject.toml ./
COPY engine/ ./engine/
COPY kb/ ./kb/
COPY services/ ./services/
COPY db/ ./db/
COPY scripts/ ./scripts/
COPY alembic.ini ./

RUN pip install -e "."

# Default command: wiki nightly trigger. Each scheduled job in
# .github/workflows/knowledge-cron.yml overrides this via
# `flyctl machine run --command "..."`.
CMD ["python", "-m", "services.synthesis.nightly_trigger"]
