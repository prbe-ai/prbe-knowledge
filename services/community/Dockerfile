FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libigraph-dev: C library required by python-igraph + leidenalg.
# build-essential: compiles native extensions. libpq-dev: asyncpg.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev curl libigraph-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY shared/ ./shared/
COPY services/ ./services/
COPY db/ ./db/
COPY scripts/ ./scripts/
COPY alembic.ini ./

RUN pip install -e "."

# Nightly Leiden community-detection cron.
# Connects to Postgres, partitions each tenant's graph, writes community_id
# back to graph_nodes, then exits cleanly. Fly runs this via [crons] at
# 03:00 UTC; see fly.community.toml.
CMD ["python", "-m", "services.community.leiden"]
