COMPOSE_FILE := deploy/annotation/docker-compose.dev.yml
ENV_FILE     := deploy/annotation/.env
ENV_EXAMPLE  := deploy/annotation/.env.dev.example

.PHONY: docker-up docker-down docker-stop docker-logs docker-status ensure-env lint type-check test-stack test test-integration test-all ci

# Compose profile options for the multi-target docker-* commands. Profiles
# determine which backing services run as containers vs. expected externally.
ALL_PROFILES := --profile all-bundled --profile external-pg --profile external-es

# Default profile for `make docker-up`. 
PROFILE ?= all-bundled

# ── Docker / Argilla stack ───────────────────────────────────────────

ensure-env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Copying $(ENV_EXAMPLE) to $(ENV_FILE)"; \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
	fi

docker-up: ensure-env ## Start Argilla stack. Override profile: make docker-up PROFILE=external-pg
# Default = all-bundled
# Override on the command line: e.g. make docker-up PROFILE=external-pg
# Valid values:
#   all-bundled  — Argilla + Postgres + Elasticsearch + Redis all in containers (default)
#   external-pg  — external Postgres (set ARGILLA_DATABASE_URL); ES + Redis bundled
#   external-es  — external Elasticsearch (set ARGILLA_ELASTICSEARCH); Postgres + Redis bundled
#   external     — all backing services external (set ARGILLA_DATABASE_URL, ARGILLA_ELASTICSEARCH, ARGILLA_REDIS_URL)
	@case "$(PROFILE)" in \
		all-bundled) PROFILE_FLAG="--profile all-bundled" ;; \
		external-pg) PROFILE_FLAG="--profile external-pg"; \
			. $(ENV_FILE) && test -n "$$ARGILLA_DATABASE_URL" || { echo "Error: ARGILLA_DATABASE_URL must be set for PROFILE=external-pg"; exit 1; } ;; \
		external-es) PROFILE_FLAG="--profile external-es"; \
			. $(ENV_FILE) && test -n "$$ARGILLA_ELASTICSEARCH" || { echo "Error: ARGILLA_ELASTICSEARCH must be set for PROFILE=external-es"; exit 1; } ;; \
		external) PROFILE_FLAG=""; \
			. $(ENV_FILE) && test -n "$$ARGILLA_DATABASE_URL" || { echo "Error: ARGILLA_DATABASE_URL must be set for PROFILE=external"; exit 1; } && \
			. $(ENV_FILE) && test -n "$$ARGILLA_ELASTICSEARCH" || { echo "Error: ARGILLA_ELASTICSEARCH must be set for PROFILE=external"; exit 1; } && \
			. $(ENV_FILE) && test -n "$$ARGILLA_REDIS_URL" || { echo "Error: ARGILLA_REDIS_URL must be set for PROFILE=external"; exit 1; } ;; \
		*) echo "Error: unknown PROFILE '$(PROFILE)' (valid: all-bundled, external-pg, external-es, external)"; exit 1 ;; \
	esac && \
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $$PROFILE_FLAG up -d --pull always --wait --remove-orphans
	@echo "Argilla is up at http://localhost:6900 (PROFILE=$(PROFILE))"

docker-down: ## Stop stack and remove volumes
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) down -v

docker-stop: ## Stop stack (preserve volumes)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) stop

docker-logs: ## Tail stack logs
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) logs -f

docker-status: ## Show stack container status
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) ps

# ── Lint & type check (mirrors CI) ──────────────────────────────────

lint: ## Run Ruff format check + linter
	uv run ruff format --check .
	uv run ruff check .

type-check: ## Run mypy type checker
	uv run mypy src/pragmata

# ── Test suites ──────────────────────────────────────────────────────

test: ## Run unit tests (default — excludes integration)
	python -m pytest

test-stack: docker-up
	@python3 -c "import urllib.request; r = urllib.request.urlopen(urllib.request.Request('http://localhost:6900/api/v1/me', headers={'X-Argilla-Api-Key': 'argilla.apikey'}), timeout=10); assert r.status == 200" || (echo "Stack health check failed" && exit 1)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) down -v
	@echo "Stack smoke test passed"

test-integration: ## Run integration tests (requires running Argilla stack)
	python -m pytest -o "addopts=" -m integration

test-all: ## Run all tests
	python -m pytest -o "addopts="

ci: lint type-check test ## Run full CI locally (lint + type-check + unit tests)
