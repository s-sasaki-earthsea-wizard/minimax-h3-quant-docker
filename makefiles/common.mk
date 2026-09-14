# Targets shared by every deployment path.
#
# The generation targets only speak HTTP to a running ComfyUI, and the scripts
# they call are standard library only, so they work identically whether the
# server came up under Docker or from a venv.

COMFYUI_REPO := https://github.com/Comfy-Org/ComfyUI.git
KJNODES_REPO := https://github.com/kijai/ComfyUI-KJNodes.git

# --- Driving a ComfyUI on another machine ------------------------------------
# REMOTE=<ssh host> sends the generation to a ComfyUI over there while the
# prompt-writing LLM stays here. This is a different axis from TARGET_ENV, which
# says how ComfyUI runs *on this machine*; the targets below are pure HTTP
# clients, so remoteness is a parameter to them rather than a second set of
# targets. It composes with either TARGET_ENV.
#
#   make pipeline THEME="..." IMAGE=still.png REMOTE=runpod-direct
#
# Setting it implies two things that are wrong to forget and silent when you do:
# the server URL moves to the forwarded port, and --upload-always goes on,
# because the data/input shortcut in prepare_image() assumes a shared filesystem
# and fails inside LoadImage at execution time when there is not one.
REMOTE_PORT ?= 9188
REMOTE_COMFY_PORT ?= $(COMFY_PORT)

ifdef REMOTE
  COMFY_SERVER := http://localhost:$(REMOTE_PORT)
  RUN := scripts/with_remote.sh $(REMOTE) $(REMOTE_PORT) $(REMOTE_COMFY_PORT)
  UPLOAD_ALWAYS := 1
else
  COMFY_SERVER := http://localhost:$(COMFY_PORT)
  RUN :=
endif

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
# KJNodes publishes no tags, so KJNODES_REF is a commit and the checkout is
# always fetch-by-sha + detach: a shallow clone alone cannot reach an arbitrary
# commit, and `pull --ff-only` would put us back on upstream HEAD.
	@if [ ! -d data/custom_nodes/ComfyUI-KJNodes/.git ]; then \
		echo ">> KJNodes: cloning"; \
		git clone --depth 1 $(KJNODES_REPO) data/custom_nodes/ComfyUI-KJNodes; \
	fi
	@echo ">> KJNodes: fetching $(KJNODES_REF)"
	@git -C data/custom_nodes/ComfyUI-KJNodes fetch --depth 1 origin $(KJNODES_REF) && \
		git -C data/custom_nodes/ComfyUI-KJNodes checkout --detach FETCH_HEAD

.PHONY: gen
gen: | .env ## Headless video: make gen PROMPT="..." [IMAGE=path] [DURATION=5] [SEED=n]
	@test -n "$(PROMPT)" || { echo 'usage: make gen PROMPT="..." [IMAGE=path] [DURATION=5] [SEED=n]'; exit 1; }
	$(RUN) python3 scripts/generate.py --prompt "$(PROMPT)" --duration "$(DURATION)" \
		$(if $(IMAGE),--image "$(IMAGE)") $(if $(UPLOAD_ALWAYS),--upload-always) \
		$(if $(SEED),--seed "$(SEED)") --server "$(COMFY_SERVER)"

.PHONY: gen-t2v
gen-t2v: | .env ## Prompt -> Ollama -> video: make gen-t2v PROMPT="..." [SPEECH=ja] [DURATION=5] [SEED=n] [MODEL=...] [DRY_RUN=1]
	@test -n "$(PROMPT)" || { echo 'usage: make gen-t2v PROMPT="..." [SPEECH=ja] [DURATION=5] [SEED=n] [MODEL=...] [DRY_RUN=1]'; exit 1; }
	@test -z "$(IMAGE)" || { echo 'gen-t2v takes no IMAGE; use: make gen-i2v IMAGE=$(IMAGE) PROMPT="..."'; exit 1; }
	$(RUN) python3 scripts/pipeline.py "$(PROMPT)" --model "$(MODEL)" --duration "$(DURATION)" \
		$(if $(SPEECH),--speech "$(SPEECH)") \
		$(if $(SEED),--seed "$(SEED)") $(if $(DRY_RUN),--dry-run) \
		--comfy-server "$(COMFY_SERVER)"

.PHONY: gen-i2v
gen-i2v: | .env ## Still + prompt -> Ollama -> video: make gen-i2v IMAGE=path [PROMPT="..."] [SPEECH=ja] [IMAGE_PROMPT="..."] [DURATION=5] [SEED=n] [DRY_RUN=1]
	@test -n "$(IMAGE)" || { echo 'usage: make gen-i2v IMAGE=path [PROMPT="..."] [SPEECH=ja] [IMAGE_PROMPT="..."] [DURATION=5] [SEED=n] [DRY_RUN=1]'; exit 1; }
	$(RUN) python3 scripts/pipeline.py $(if $(PROMPT),"$(PROMPT)") --image "$(IMAGE)" \
		--model "$(MODEL)" --duration "$(DURATION)" \
		$(if $(UPLOAD_ALWAYS),--upload-always) \
		$(if $(IMAGE_PROMPT),--image-prompt "$(IMAGE_PROMPT)") \
		$(if $(SPEECH),--speech "$(SPEECH)") \
		$(if $(SEED),--seed "$(SEED)") $(if $(DRY_RUN),--dry-run) \
		--comfy-server "$(COMFY_SERVER)"

.PHONY: pipeline
pipeline: | .env ## Same, with the mode taken from IMAGE=: make pipeline THEME="..." [IMAGE=path] [SPEECH=ja] ...
	@test -n "$(THEME)$(IMAGE)" || { echo 'usage: make pipeline THEME="..." [IMAGE=path] [IMAGE_PROMPT="..."] [SPEECH=ja] [MODEL=...] [DURATION=5] [SEED=n] [DRY_RUN=1]'; exit 1; }
	$(RUN) python3 scripts/pipeline.py $(if $(THEME),"$(THEME)") --model "$(MODEL)" --duration "$(DURATION)" \
		$(if $(IMAGE),--image "$(IMAGE)") $(if $(UPLOAD_ALWAYS),--upload-always) \
		$(if $(IMAGE_PROMPT),--image-prompt "$(IMAGE_PROMPT)") \
		$(if $(SPEECH),--speech "$(SPEECH)") \
		$(if $(SEED),--seed "$(SEED)") $(if $(DRY_RUN),--dry-run) \
		--comfy-server "$(COMFY_SERVER)"

.PHONY: nvidia-smi
nvidia-smi: ## Show GPU usage
	@nvidia-smi
