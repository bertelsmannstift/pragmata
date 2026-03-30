COMPOSE_FILE := deploy/annotation/docker-compose.dev.yml
ENV_FILE     := deploy/annotation/.env
ENV_EXAMPLE  := deploy/annotation/.env.dev.example

.PHONY: docker-up docker-up-external-pg docker-up-external-es docker-up-external docker-down docker-stop docker-logs docker-status ensure-env test-stack test test-integration test-all

# Profiles: all-bundled (default), external-pg (ext Postgres), external-es (ext ES), no profile (all external)
ALL_PROFILES := --profile all-bundled --profile external-pg --profile external-es

# ── Docker / Argilla stack ───────────────────────────────────────────

ensure-env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Copying $(ENV_EXAMPLE) to $(ENV_FILE)"; \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
	fi

docker-up: ensure-env ## Start Argilla stack (all services bundled)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile all-bundled up -d --pull always --wait
	@echo "Argilla is up at http://localhost:6900"

docker-up-external-pg: ensure-env ## Start Argilla stack with external Postgres (ES + Redis bundled)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile external-pg up -d --pull always --wait
	@echo "Argilla is up at http://localhost:6900 (external Postgres, bundled ES + Redis)"

docker-up-external-es: ensure-env ## Start Argilla stack with external Elasticsearch (Postgres + Redis bundled)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile external-es up -d --pull always --wait
	@echo "Argilla is up at http://localhost:6900 (external Elasticsearch, bundled Postgres + Redis)"

docker-up-external: ensure-env ## Start Argilla stack with all external backing services
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --pull always --wait --remove-orphans
	@echo "Argilla is up at http://localhost:6900 (all external backing services)"

docker-down: ## Stop stack and remove volumes
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) down -v

docker-stop: ## Stop stack (preserve volumes)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) stop

docker-logs: ## Tail stack logs
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) logs -f

docker-status: ## Show stack container status
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) ps

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
