COMPOSE_FILE := deploy/annotation/docker-compose.dev.yml
ENV_FILE     := deploy/annotation/.env
ENV_EXAMPLE  := deploy/annotation/.env.dev.example

.PHONY: setup teardown stop logs status test-stack

setup:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Copying $(ENV_EXAMPLE) to $(ENV_FILE)"; \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
	fi
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d --pull always --wait
	@echo "Argilla is up at http://localhost:6900"

teardown:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) down -v

stop:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) stop

logs:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) logs -f

status:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) ps

test-stack: setup
	@curl -sf http://localhost:6900/api/v1/status | grep -q '"status"' || (echo "Stack health check failed" && exit 1)
	$(MAKE) teardown
	@echo "Stack smoke test passed"
