# Targets shared by every deployment path.
#
# The generation targets only speak HTTP to a running ComfyUI, and the scripts
# they call are standard library only, so they work identically whether the
# server came up under Docker or from a venv.

COMFYUI_REPO := https://github.com/Comfy-Org/ComfyUI.git
KJNODES_REPO := https://github.com/kijai/ComfyUI-KJNodes.git

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

.PHONY: gen
gen: | .env ## Headless video: make gen PROMPT="..." [IMAGE=path] [DURATION=5] [SEED=n]
	@test -n "$(PROMPT)" || { echo 'usage: make gen PROMPT="..." [IMAGE=path] [DURATION=5] [SEED=n]'; exit 1; }
	python3 scripts/generate.py --prompt "$(PROMPT)" --duration "$(DURATION)" \
		$(if $(IMAGE),--image "$(IMAGE)") \
		$(if $(SEED),--seed "$(SEED)") --server "http://localhost:$(COMFY_PORT)"

.PHONY: gen-t2v
gen-t2v: | .env ## Prompt -> Ollama -> video: make gen-t2v PROMPT="..." [SPEECH=ja] [DURATION=5] [SEED=n] [MODEL=...] [DRY_RUN=1]
	@test -n "$(PROMPT)" || { echo 'usage: make gen-t2v PROMPT="..." [SPEECH=ja] [DURATION=5] [SEED=n] [MODEL=...] [DRY_RUN=1]'; exit 1; }
	@test -z "$(IMAGE)" || { echo 'gen-t2v takes no IMAGE; use: make gen-i2v IMAGE=$(IMAGE) PROMPT="..."'; exit 1; }
	python3 scripts/pipeline.py "$(PROMPT)" --model "$(MODEL)" --duration "$(DURATION)" \
		$(if $(SPEECH),--speech "$(SPEECH)") \
		$(if $(SEED),--seed "$(SEED)") $(if $(DRY_RUN),--dry-run) \
		--comfy-server "http://localhost:$(COMFY_PORT)"

.PHONY: gen-i2v
gen-i2v: | .env ## Still + prompt -> Ollama -> video: make gen-i2v IMAGE=path [PROMPT="..."] [SPEECH=ja] [IMAGE_PROMPT="..."] [DURATION=5] [SEED=n] [DRY_RUN=1]
	@test -n "$(IMAGE)" || { echo 'usage: make gen-i2v IMAGE=path [PROMPT="..."] [SPEECH=ja] [IMAGE_PROMPT="..."] [DURATION=5] [SEED=n] [DRY_RUN=1]'; exit 1; }
	python3 scripts/pipeline.py $(if $(PROMPT),"$(PROMPT)") --image "$(IMAGE)" \
		--model "$(MODEL)" --duration "$(DURATION)" \
		$(if $(IMAGE_PROMPT),--image-prompt "$(IMAGE_PROMPT)") \
		$(if $(SPEECH),--speech "$(SPEECH)") \
		$(if $(SEED),--seed "$(SEED)") $(if $(DRY_RUN),--dry-run) \
		--comfy-server "http://localhost:$(COMFY_PORT)"

.PHONY: pipeline
pipeline: | .env ## Same, with the mode taken from IMAGE=: make pipeline THEME="..." [IMAGE=path] [SPEECH=ja] ...
	@test -n "$(THEME)$(IMAGE)" || { echo 'usage: make pipeline THEME="..." [IMAGE=path] [IMAGE_PROMPT="..."] [SPEECH=ja] [MODEL=...] [DURATION=5] [SEED=n] [DRY_RUN=1]'; exit 1; }
	python3 scripts/pipeline.py $(if $(THEME),"$(THEME)") --model "$(MODEL)" --duration "$(DURATION)" \
		$(if $(IMAGE),--image "$(IMAGE)") \
		$(if $(IMAGE_PROMPT),--image-prompt "$(IMAGE_PROMPT)") \
		$(if $(SPEECH),--speech "$(SPEECH)") \
		$(if $(SEED),--seed "$(SEED)") $(if $(DRY_RUN),--dry-run) \
		--comfy-server "http://localhost:$(COMFY_PORT)"

.PHONY: nvidia-smi
nvidia-smi: ## Show GPU usage
	@nvidia-smi
