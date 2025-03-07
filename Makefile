.PHONY: build start stop logs ps status clean build-images build-ui build-admin-ui help

# Build and start everything
all: build start

# Help command
help:
	@echo "Llora Lab Makefile"
	@echo "===================="
	@echo ""
	@echo "Commands:"
	@echo "  make build        Build admin-api, trainer, and vllm images"
	@echo "  make build-ui     Build the admin UI frontend"
	@echo "  make start        Start the admin containers"
	@echo "  make stop         Stop all containers"
	@echo "  make logs         View logs (make logs service=admin-api)"
	@echo "  make ps           List running containers"
	@echo "  make clean        Remove all containers and volumes"
	@echo "  make build-images Build only the pre-built images"
	@echo "  make help         Show this help"

# Build all images except admin-ui
build:
	@echo "Building API, trainer and vLLM images..."
	docker compose build admin-api trainer vllm

# Build admin UI separately
build-ui:
	@echo "Building admin UI..."
	cd admin-ui && npm install && npm run build

# Build admin-ui container (requires build-ui first)
build-admin-ui: build-ui
	@echo "Building admin UI container..."
	docker compose build admin-ui

# Build only prebuilt images (trainer and vllm)
build-images:
	@echo "Building trainer and vLLM images only..."
	docker compose build trainer vllm

# Start the admin services
start:
	@echo "Starting admin services..."
	docker compose up -d admin-api admin-ui

# Stop all services
stop:
	@echo "Stopping all services..."
	docker compose down

# View logs (usage: make logs service=admin-api)
logs:
	@if [ -z "$(service)" ]; then \
		docker compose logs -f; \
	else \
		docker compose logs -f $(service); \
	fi

# Show running containers
ps:
	docker compose ps

# Show status of components
status:
	@echo "Llora Lab Status"
	@echo "================="
	@echo ""
	@echo "Services:"
	@docker compose ps | grep -v "^Name\|^-------" | awk '{ printf "  %-20s %s\n", $$1, $$4 }'
	@echo ""
	@echo "Disk Usage:"
	@du -sh ./huggingface-cache ./adapters ./datasets ./logs | awk '{ printf "  %-20s %s\n", $$2, $$1 }'

# Clean up everything
clean:
	@echo "Stopping all containers..."
	docker compose down -v
	@echo "Done."

# Development utilities
jupyter:
	docker compose run --rm --service-ports trainer jupyter lab --ip=0.0.0.0 --allow-root --no-browser

