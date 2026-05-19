COMPOSE_SHIPPED  := src/pragmata/annotation/docker-compose.yml
COMPOSE_OVERRIDE := deploy/annotation/docker-compose.dev.override.yml
COMPOSE_FILES    := -f $(COMPOSE_SHIPPED) -f $(COMPOSE_OVERRIDE)
ENV_FILE         := deploy/annotation/.env
ENV_EXAMPLE      := deploy/annotation/.env.dev.example

.PHONY: docker-up docker-up-external-pg docker-up-external-es docker-up-external-redis docker-up-external docker-down docker-stop docker-logs docker-status ensure-env lint type-check test-stack test test-integration test-all ci

# Profiles: all-bundled (default), external-pg / external-es / external-redis (one
# backing service external), no profile (all external). Listing every profile in
# ALL_PROFILES makes down / stop / logs / status catch containers regardless of
# which profile was used to bring them up.
ALL_PROFILES := --profile all-bundled --profile external-pg --profile external-es --profile external-redis

# ── Docker / Argilla stack ───────────────────────────────────────────

ensure-env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Copying $(ENV_EXAMPLE) to $(ENV_FILE)"; \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
	fi

docker-up: ensure-env ## Start Argilla stack (all services bundled)
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) --profile all-bundled up -d --pull always --wait --remove-orphans
	@echo "Argilla is up at http://localhost:6900"

docker-up-external-pg: ensure-env ## Start Argilla stack with external Postgres (ES + Redis bundled)
	@. $(ENV_FILE) && test -n "$$ARGILLA_DATABASE_URL" || { echo "Error: ARGILLA_DATABASE_URL must be set for external Postgres"; exit 1; }
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) --profile external-pg up -d --pull always --wait --remove-orphans
	@echo "Argilla is up at http://localhost:6900 (external Postgres, bundled ES + Redis)"

docker-up-external-es: ensure-env ## Start Argilla stack with external Elasticsearch (Postgres + Redis bundled)
	@. $(ENV_FILE) && test -n "$$ARGILLA_ELASTICSEARCH" || { echo "Error: ARGILLA_ELASTICSEARCH must be set for external Elasticsearch"; exit 1; }
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) --profile external-es up -d --pull always --wait --remove-orphans
	@echo "Argilla is up at http://localhost:6900 (external Elasticsearch, bundled Postgres + Redis)"

docker-up-external-redis: ensure-env ## Start Argilla stack with external Redis (Postgres + ES bundled)
	@. $(ENV_FILE) && test -n "$$ARGILLA_REDIS_URL" || { echo "Error: ARGILLA_REDIS_URL must be set for external Redis"; exit 1; }
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) --profile external-redis up -d --pull always --wait --remove-orphans
	@echo "Argilla is up at http://localhost:6900 (external Redis, bundled Postgres + ES)"

docker-up-external: ensure-env ## Start Argilla stack with all external backing services
	@. $(ENV_FILE) && test -n "$$ARGILLA_DATABASE_URL" || { echo "Error: ARGILLA_DATABASE_URL must be set for external Postgres"; exit 1; }
	@. $(ENV_FILE) && test -n "$$ARGILLA_ELASTICSEARCH" || { echo "Error: ARGILLA_ELASTICSEARCH must be set for external Elasticsearch"; exit 1; }
	@. $(ENV_FILE) && test -n "$$ARGILLA_REDIS_URL" || { echo "Error: ARGILLA_REDIS_URL must be set for external Redis"; exit 1; }
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) up -d --pull always --wait --remove-orphans
	@echo "Argilla is up at http://localhost:6900 (all external backing services)"

docker-down: ## Stop stack and remove volumes
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) $(ALL_PROFILES) down -v

docker-stop: ## Stop stack (preserve volumes)
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) $(ALL_PROFILES) stop

docker-logs: ## Tail stack logs
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) $(ALL_PROFILES) logs -f

docker-status: ## Show stack container status
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) $(ALL_PROFILES) ps

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
	docker compose $(COMPOSE_FILES) --env-file $(ENV_FILE) $(ALL_PROFILES) down -v
	@echo "Stack smoke test passed"

test-integration: ## Run integration tests (requires running Argilla stack)
	python -m pytest -o "addopts=" -m integration

test-all: ## Run all tests
	python -m pytest -o "addopts="

ci: lint type-check test ## Run full CI locally (lint + type-check + unit tests)
