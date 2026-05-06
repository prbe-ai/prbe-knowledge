FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY shared/ ./shared/
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
