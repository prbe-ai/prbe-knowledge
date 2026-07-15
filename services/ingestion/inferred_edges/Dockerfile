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
COPY engine/ ./engine/
COPY kb/ ./kb/
COPY services/ ./services/
COPY db/ ./db/
COPY alembic.ini ./

RUN pip install -e "."

# Health endpoint: 0.0.0.0 (IPv4) for Fly http_service.checks
# (feedback_fly_health_check_ipv4.md).
# The worker itself does not serve HTTP traffic other than /health.
EXPOSE 8083

CMD python -m services.ingestion.inferred_edges.worker
