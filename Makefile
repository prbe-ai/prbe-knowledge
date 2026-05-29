# Probe knowledge engine — self-host quickstart.
#
#   cp .env.example .env   # fill GOOGLE_API_KEY + one LLM key + KNOWLEDGE_API_TOKEN
#   make up                # build + start the whole stack
#   make health            # confirm services are up
#   make query Q="what changed last week?"

COMPOSE ?= docker compose
TOKEN   ?= $(shell grep -E '^KNOWLEDGE_API_TOKEN=' .env 2>/dev/null | cut -d= -f2)

.PHONY: up down logs ps health migrate seed query

up:            ## Build images and start the full stack
	$(COMPOSE) up -d --build

down:          ## Stop the stack (keeps volumes)
	$(COMPOSE) down

logs:          ## Tail logs for all services
	$(COMPOSE) logs -f

ps:            ## Show service status
	$(COMPOSE) ps

migrate:       ## Re-run the DB migration (idempotent)
	$(COMPOSE) run --rm migrate

health:        ## Hit /health on ingestion + retrieval
	@echo "ingestion:" && curl -fsS http://localhost:8080/health && echo
	@echo "retrieval:" && curl -fsS http://localhost:8081/health && echo

# Ingest a sample doc via the custom-ingest API. In single-tenant self-host mode
# this authenticates with the KNOWLEDGE_API_TOKEN bearer (same token as
# `make query`); a hosted gateway deployment uses X-Internal-Knowledge-Key.
seed:          ## Ingest one sample doc for the default tenant
	curl -fsS -X POST http://localhost:8080/api/custom-ingest/documents \
	  -H "Authorization: Bearer $(TOKEN)" \
	  -H "Content-Type: application/json" \
	  -d '{"source_key":"seed","documents":[{"id":"seed-1","body":"Probe is a self-hosted knowledge engine. This is a seed document."}]}' \
	  && echo

# Run a query against the retrieval service (needs GOOGLE_API_KEY for embeddings).
query:         ## Query: make query Q="your question"
	curl -fsS -X POST http://localhost:8081/query \
	  -H "Authorization: Bearer $(TOKEN)" \
	  -H "Content-Type: application/json" \
	  -d "{\"query\": \"$(Q)\"}" \
	  && echo
