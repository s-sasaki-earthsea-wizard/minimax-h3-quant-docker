# Docker + Compose path -- the original one, validated on the RTX 5080 box.
#
# ComfyUI runs in a container built from docker/Dockerfile; the checkout and
# everything stateful are bind-mounted from the host.

COMPOSE := docker compose -f docker/compose.yaml --env-file .env

.PHONY: setup
setup: checkout build | .env ## First-time setup (fetch sources + build image)
	@echo
	@echo "Next: run 'make models' to download the 42.5GB weights."

.PHONY: build
build: | .env ## Build the Docker image
	$(COMPOSE) build

.PHONY: models
models: | .env ## Download weights (minimal FL2VA set, 42.5GB)
	$(COMPOSE) run --rm --no-deps -v "$(CURDIR)/scripts:/scripts:ro" \
		--entrypoint bash comfyui /scripts/download_models.sh

.PHONY: models-ref2va
models-ref2va: | .env ## Additionally fetch the Ref2VA DiT (+21GB)
	$(COMPOSE) run --rm --no-deps -v "$(CURDIR)/scripts:/scripts:ro" \
		-e TASKS=ref2va --entrypoint bash comfyui /scripts/download_models.sh

.PHONY: models-turbo
models-turbo: | .env ## Additionally fetch the 4/8-step turbo LoRAs (+3.9GB)
	$(COMPOSE) run --rm --no-deps -v "$(CURDIR)/scripts:/scripts:ro" \
		-e TURBO=1 --entrypoint bash comfyui /scripts/download_models.sh

.PHONY: up
up: | .env ## Start ComfyUI (background)
	$(COMPOSE) up -d
	@echo
	@echo "  http://localhost:$(COMFY_PORT)"
	@echo "  Logs: make logs"

.PHONY: down
down: | .env ## Stop and remove containers
	$(COMPOSE) down

.PHONY: status
status: | .env ## Report whether ComfyUI is up
	@$(COMPOSE) ps

.PHONY: logs
logs: | .env ## Follow the logs
	$(COMPOSE) logs -f

.PHONY: shell
shell: | .env ## Open a bash shell in the container
	$(COMPOSE) run --rm --no-deps --entrypoint bash comfyui

.PHONY: doctor
doctor: | .env ## Verify GPU / torch / quantization path / nodes
	$(COMPOSE) run --rm --no-deps -v "$(CURDIR)/scripts:/scripts:ro" \
		--entrypoint python comfyui /scripts/doctor.py

.PHONY: clean
clean: | .env ## Remove containers and images (weights are kept)
	$(COMPOSE) down --rmi local
