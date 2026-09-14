# Plain-venv path, for hosts where Docker is unavailable rather than merely
# unwanted.
#
# A RunPod GPU pod is the case this was written for: the pod is itself a
# container running with the default capability set and no /dev/fuse, so neither
# Docker-in-Docker nor rootless Podman can work there. The pins, the layout and
# the workflow templates are identical to the onprem path -- only the isolation
# mechanism is missing.
#
# The venv lives outside the working tree: it is ~20GB of build output, it is
# not in .gitignore, and it should survive re-cloning the repo.
VENV ?= $(CURDIR)/../venv
VENV_PY := $(VENV)/bin/python

.PHONY: setup
setup: checkout venv | .env ## First-time setup (fetch sources + build venv)
	@echo
	@echo "Next: run 'make models' to download the 42.5GB weights."

.PHONY: venv
venv: | .env ## Build the venv (torch cu130, deps, SageAttention from source)
	VENV="$(VENV)" scripts/setup_native.sh

.PHONY: models
models: ## Download weights (minimal FL2VA set, 42.5GB)
	PATH="$(VENV)/bin:$$PATH" MODELS_DIR="$(CURDIR)/data/models" \
		scripts/download_models.sh

.PHONY: models-ref2va
models-ref2va: ## Additionally fetch the Ref2VA DiT (+21GB)
	PATH="$(VENV)/bin:$$PATH" MODELS_DIR="$(CURDIR)/data/models" TASKS=ref2va \
		scripts/download_models.sh

.PHONY: up
up: | .env ## Start ComfyUI (background, survives SSH loss)
	VENV="$(VENV)" scripts/run_native.sh start

.PHONY: down
down: ## Stop ComfyUI
	VENV="$(VENV)" scripts/run_native.sh stop

.PHONY: status
status: ## Report whether ComfyUI is up
	@VENV="$(VENV)" scripts/run_native.sh status

.PHONY: logs
logs: ## Follow the logs
	VENV="$(VENV)" scripts/run_native.sh logs

.PHONY: shell
shell: ## Open a bash shell with the venv activated
	@PATH="$(VENV)/bin:$$PATH" VIRTUAL_ENV="$(VENV)" bash

.PHONY: doctor
doctor: ## Verify GPU / torch / quantization path / nodes
	COMFYUI_DIR="$(CURDIR)/ComfyUI" $(VENV_PY) scripts/doctor.py

.PHONY: clean
clean: ## Remove the venv (weights and checkouts are kept)
	rm -rf "$(VENV)"
