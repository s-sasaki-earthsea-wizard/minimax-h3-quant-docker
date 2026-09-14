SHELL := /bin/bash

# .env is gitignored, so on a fresh clone it does not exist yet. Variables are
# expanded while the makefile is parsed -- before the .env rule below could run
# -- so read from .env.example whenever .env is still missing.
ENV_SRC := $(if $(wildcard .env),.env,.env.example)
env_get = $(shell grep -E '^$(1)=' $(ENV_SRC) | cut -d= -f2)

COMFYUI_REF := $(call env_get,COMFYUI_REF)
KJNODES_REF := $(call env_get,KJNODES_REF)
COMFY_PORT := $(call env_get,COMFY_PORT)

# Which deployment path this machine uses -- see makefiles/.
#   onprem  Docker + Compose
#   cloud   plain venv, for hosts where Docker cannot run at all
# It lives in .env because .env is already the per-machine, gitignored config
# file, so the machine that needs a different path is the machine that already
# has its own copy. .env.example ships the onprem default.
TARGET_ENV := $(or $(call env_get,TARGET_ENV),onprem)

.DEFAULT_GOAL := help

# Headless generation defaults (override per-invocation: make gen DURATION=10)
DURATION ?= 5
MODEL ?= hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q4_K_M

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
	@echo ">> TARGET_ENV=$(TARGET_ENV) -- edit .env if this machine cannot run Docker"

.PHONY: help
help: ## Show this help
	@printf '  target env: \033[36m%s\033[0m  (makefiles/%s.mk -- change TARGET_ENV in .env)\n\n' \
		"$(TARGET_ENV)" "$(TARGET_ENV)"
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# Targets that do not care which path is in use.
include makefiles/common.mk

# Fail with an explanation rather than make's bare "No such file or directory".
ifeq ($(wildcard makefiles/$(TARGET_ENV).mk),)
$(error TARGET_ENV='$(TARGET_ENV)' has no makefiles/$(TARGET_ENV).mk. \
        Set TARGET_ENV=onprem or TARGET_ENV=cloud in .env)
endif
include makefiles/$(TARGET_ENV).mk
