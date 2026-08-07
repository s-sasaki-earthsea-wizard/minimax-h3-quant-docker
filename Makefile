SHELL := /bin/bash
COMPOSE := docker compose -f docker/compose.yaml --env-file .env
COMFYUI_REPO := https://github.com/Comfy-Org/ComfyUI.git
KJNODES_REPO := https://github.com/kijai/ComfyUI-KJNodes.git

# .env is gitignored, so on a fresh clone it does not exist yet. Variables are
# expanded while the makefile is parsed -- before the .env rule below could run
# -- so read from .env.example whenever .env is still missing.
ENV_SRC := $(if $(wildcard .env),.env,.env.example)
env_get = $(shell grep -E '^$(1)=' $(ENV_SRC) | cut -d= -f2)
COMFYUI_REF := $(call env_get,COMFYUI_REF)
COMFY_PORT := $(call env_get,COMFY_PORT)

.DEFAULT_GOAL := help

# Materialise .env on first use, with this machine's uid/gid rather than the
# 1000/1000 placeholder -- the container user is created from them, so a
# mismatch leaves bind-mounted files unwritable. The prerequisite is order-only
# on purpose: a `git pull` that touches .env.example must never overwrite the
# .env you have been editing.
.env: | .env.example
	@cp .env.example $@
	@sed -i -e "s/^HOST_UID=.*/HOST_UID=$$(id -u)/" \
	        -e "s/^HOST_GID=.*/HOST_GID=$$(id -g)/" $@
	@echo ">> created .env from .env.example (uid=$$(id -u) gid=$$(id -g))"

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: checkout build | .env ## First-time setup (fetch sources + build image)
	@echo
	@echo "Next: run 'make models' to download the 42.5GB weights."

.PHONY: checkout
checkout: ## Fetch / update ComfyUI and KJNodes
	@if [ -d ComfyUI/.git ]; then \
		echo ">> ComfyUI: fetching $(COMFYUI_REF)"; \
		git -C ComfyUI fetch --tags --depth 1 origin $(COMFYUI_REF) && \
		git -C ComfyUI checkout --detach FETCH_HEAD; \
	else \
		echo ">> ComfyUI: cloning $(COMFYUI_REF)"; \
		git clone --depth 1 --branch $(COMFYUI_REF) $(COMFYUI_REPO) ComfyUI; \
	fi
	@if [ -d data/custom_nodes/ComfyUI-KJNodes/.git ]; then \
		echo ">> KJNodes: pulling"; \
		git -C data/custom_nodes/ComfyUI-KJNodes pull --ff-only; \
	else \
		echo ">> KJNodes: cloning"; \
		git clone --depth 1 $(KJNODES_REPO) data/custom_nodes/ComfyUI-KJNodes; \
	fi

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

.PHONY: up
up: | .env ## Start ComfyUI (background)
	$(COMPOSE) up -d
	@echo
	@echo "  http://localhost:$(COMFY_PORT)"
	@echo "  Logs: make logs"

.PHONY: down
down: | .env ## Stop and remove containers
	$(COMPOSE) down

.PHONY: logs
logs: | .env ## Follow the logs
	$(COMPOSE) logs -f

.PHONY: shell
shell: | .env ## Open a bash shell in the container
	$(COMPOSE) run --rm --no-deps --entrypoint bash comfyui

.PHONY: doctor
doctor: | .env ## Verify GPU / torch / quantization path / nodes from inside the container
	$(COMPOSE) run --rm --no-deps -v "$(CURDIR)/scripts:/scripts:ro" \
		--entrypoint python comfyui /scripts/doctor.py

.PHONY: nvidia-smi
nvidia-smi: ## Show GPU usage
	@nvidia-smi

.PHONY: clean
clean: | .env ## Remove containers and images (weights are kept)
	$(COMPOSE) down --rmi local
