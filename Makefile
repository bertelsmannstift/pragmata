COMPOSE_FILE := deploy/annotation/docker-compose.dev.yml
ENV_FILE     := deploy/annotation/.env
ENV_EXAMPLE  := deploy/annotation/.env.dev.example

.PHONY: docker-up docker-up-external docker-down docker-stop docker-logs docker-status test-stack test test-integration test-all

# ── Docker / Argilla stack ───────────────────────────────────────────

docker-up: ## Start Argilla stack (all services bundled)
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Copying $(ENV_EXAMPLE) to $(ENV_FILE)"; \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
	fi
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile bundled up -d --pull always --wait
	@echo "Argilla is up at http://localhost:6900"

docker-up-external: ## Start Argilla stack with external backing services
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Copying $(ENV_EXAMPLE) to $(ENV_FILE)"; \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
	fi
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --pull always --wait
	@echo "Argilla is up (using external backing services)"

docker-down: ## Stop stack and remove volumes
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile bundled down -v

docker-stop: ## Stop stack (preserve volumes)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile bundled stop

docker-logs: ## Tail stack logs
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile bundled logs -f

docker-status: ## Show stack container status
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile bundled ps

# ── Test suites ──────────────────────────────────────────────────────

test: ## Run unit tests (default — excludes integration)
	python -m pytest

test-stack: docker-up
	@python3 -c "import urllib.request; r = urllib.request.urlopen(urllib.request.Request('http://localhost:6900/api/v1/me', headers={'X-Argilla-Api-Key': 'argilla.apikey'}), timeout=10); assert r.status == 200" || (echo "Stack health check failed" && exit 1)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile bundled down -v
	@echo "Stack smoke test passed"

test-integration: ## Run integration tests (requires running Argilla stack)
	python -m pytest -o "addopts=" -m integration

test-all: ## Run all tests
	python -m pytest -o "addopts="
