COMPOSE_FILE := deploy/annotation/docker-compose.dev.yml
ENV_FILE     := deploy/annotation/.env
ENV_EXAMPLE  := deploy/annotation/.env.dev.example

# One throwaway stack for all test targets — an isolated Compose project on a
# non-default host port so its destructive setup/teardown can never touch the
# live stack (Compose project 'annotation') on :6900. The -p project scope
# always wins over the shared compose file, so containers/volumes stay disjoint.
# test-stack and test-integration/test-all share it (they never run at once).
TEST_STACK_PROJECT := pragmata-test
TEST_STACK_PORT    := 6901

.PHONY: docker-up docker-up-external-pg docker-up-external-es docker-up-external docker-down docker-down-clean docker-stop docker-logs docker-status ensure-env lint type-check test-stack test-unit test-integration test-all ci

# Profiles: all-bundled (default), external-pg (ext Postgres), external-es (ext ES), no profile (all external)
ALL_PROFILES := --profile all-bundled --profile external-pg --profile external-es

# ── Docker / Argilla stack ───────────────────────────────────────────

ensure-env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Copying $(ENV_EXAMPLE) to $(ENV_FILE)"; \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
	fi

docker-up: ensure-env ## Start Argilla stack (all services bundled)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile all-bundled up -d --pull always --wait --remove-orphans
	@echo "Argilla is up at http://localhost:6900"

docker-up-external-pg: ensure-env ## Start Argilla stack with external Postgres (ES + Redis bundled)
	@. $(ENV_FILE) && test -n "$$ARGILLA_DATABASE_URL" || { echo "Error: ARGILLA_DATABASE_URL must be set for external Postgres"; exit 1; }
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile external-pg up -d --pull always --wait --remove-orphans
	@echo "Argilla is up at http://localhost:6900 (external Postgres, bundled ES + Redis)"

docker-up-external-es: ensure-env ## Start Argilla stack with external Elasticsearch (Postgres + Redis bundled)
	@. $(ENV_FILE) && test -n "$$ARGILLA_ELASTICSEARCH" || { echo "Error: ARGILLA_ELASTICSEARCH must be set for external Elasticsearch"; exit 1; }
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile external-es up -d --pull always --wait --remove-orphans
	@echo "Argilla is up at http://localhost:6900 (external Elasticsearch, bundled Postgres + Redis)"

docker-up-external: ensure-env ## Start Argilla stack with all external backing services
	@. $(ENV_FILE) && test -n "$$ARGILLA_DATABASE_URL" || { echo "Error: ARGILLA_DATABASE_URL must be set for external Postgres"; exit 1; }
	@. $(ENV_FILE) && test -n "$$ARGILLA_ELASTICSEARCH" || { echo "Error: ARGILLA_ELASTICSEARCH must be set for external Elasticsearch"; exit 1; }
	@. $(ENV_FILE) && test -n "$$ARGILLA_REDIS_URL" || { echo "Error: ARGILLA_REDIS_URL must be set for external Redis"; exit 1; }
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --pull always --wait --remove-orphans
	@echo "Argilla is up at http://localhost:6900 (all external backing services)"

docker-down: ## Stop stack, preserve volumes (annotation data kept)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) down

docker-down-clean: ## Stop stack and delete all volumes (annotation data lost — irreversible)
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) down -v

docker-stop: ## Pause stack without removing containers (preserve volumes)
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
#
#   test-unit         unit tests only — no stack, fast (default dev loop)
#   test-integration  integration tests on a throwaway isolated stack
#   test-all          every test (unit + integration + packaging) on that stack
#   test-stack        stack health smoke test (isolated project + port)
#
# test-integration / test-all stand up an ephemeral Argilla on :$(TEST_STACK_PORT)
# under an isolated Compose project, point the tests at it via
# PRAGMATA_TEST_ARGILLA_URL, and always tear it down (volumes included). The live
# stack on :6900 is never touched. Integration tests also
# refuse to run unless PRAGMATA_TEST_ARGILLA_URL is set (tests/conftest.py), so a
# bare `pytest -m integration` can't fall back to :6900 either.

test-unit: ## Run unit tests only — no stack (fast default loop)
	python -m pytest

test-integration: PYTEST_SELECT := -m integration
test-all:         PYTEST_SELECT :=
test-integration test-all: ensure-env
	@set -e; \
	export ARGILLA_PORT=$(TEST_STACK_PORT); \
	trap 'docker compose -p $(TEST_STACK_PROJECT) -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) down -v >/dev/null 2>&1' EXIT; \
	docker compose -p $(TEST_STACK_PROJECT) -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile all-bundled up -d --pull always --wait --remove-orphans; \
	API_KEY=$$(. $(ENV_FILE); echo $${ARGILLA_API_KEY:-argilla.apikey}); \
	PRAGMATA_TEST_ARGILLA_URL=http://localhost:$(TEST_STACK_PORT) \
	PRAGMATA_TEST_ARGILLA_API_KEY=$$API_KEY \
		python -m pytest -o "addopts=" $(PYTEST_SELECT)

test-stack: ensure-env ## Smoke-test the stack in an isolated Compose project + port (never touches :6900)
	@set -e; \
	export ARGILLA_PORT=$(TEST_STACK_PORT); \
	trap 'docker compose -p $(TEST_STACK_PROJECT) -f $(COMPOSE_FILE) --env-file $(ENV_FILE) $(ALL_PROFILES) down -v >/dev/null 2>&1' EXIT; \
	docker compose -p $(TEST_STACK_PROJECT) -f $(COMPOSE_FILE) --env-file $(ENV_FILE) --profile all-bundled up -d --pull always --wait --remove-orphans; \
	API_KEY=$$(. $(ENV_FILE); echo $${ARGILLA_API_KEY:-argilla.apikey}); \
	python3 -c "import urllib.request; r = urllib.request.urlopen(urllib.request.Request('http://localhost:$(TEST_STACK_PORT)/api/v1/me', headers={'X-Argilla-Api-Key': '$$API_KEY'}), timeout=10); assert r.status == 200" \
		|| { echo 'Stack health check failed'; exit 1; }; \
	echo "Stack smoke test passed (isolated project '$(TEST_STACK_PROJECT)' on :$(TEST_STACK_PORT), torn down + volumes cleaned)"

ci: lint type-check test-unit ## Run full CI locally (lint + type-check + unit tests)
