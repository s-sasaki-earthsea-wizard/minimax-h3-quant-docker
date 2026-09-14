#!/usr/bin/env bash
#
# Build the pinned ComfyUI stack directly in a venv, with no Docker.
#
# For hosts where Docker is unavailable rather than merely unwanted -- RunPod
# GPU pods, for instance, are themselves containers with the default capability
# set, so neither Docker-in-Docker nor rootless Podman can work there.
#
# This is the same recipe as docker/Dockerfile, minus the parts the image only
# needs because it starts from a bare base: the CUDA toolchain and the Python
# runtime are expected to be present already. Versions are read from .env (or
# .env.example) so the pins stay single-sourced with the Docker path.
#
# Requires: Ubuntu 24.04, CUDA 13.0 toolkit, Python 3.12, a Blackwell GPU.
#
# Sources are fetched by the shared `checkout` make target, not here -- run
# `make setup` (which is `checkout` then `venv`) rather than calling this
# directly on a fresh clone.
#
# Usage:
#   scripts/setup_native.sh              # build into $REPO/../venv
#   VENV=/somewhere scripts/setup_native.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$(dirname "${REPO_ROOT}")/venv}"

# Keep the venv outside the working tree: it is 20 GB of build output, it is not
# in .gitignore, and it should survive re-cloning the repo.

log() { printf '\n>> %s\n' "$*"; }
die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }

# --- Pins, read from the same file the Docker path uses -----------------------
ENV_SRC="${REPO_ROOT}/.env"
[ -f "${ENV_SRC}" ] || ENV_SRC="${REPO_ROOT}/.env.example"
[ -f "${ENV_SRC}" ] || die "neither .env nor .env.example found in ${REPO_ROOT}"

env_get() { grep -E "^$1=" "${ENV_SRC}" | head -1 | cut -d= -f2-; }
TORCH_VERSION="$(env_get TORCH_VERSION)"
SAGEATTENTION_REF="$(env_get SAGEATTENTION_REF)"
[ -n "${TORCH_VERSION}" ] && [ -n "${SAGEATTENTION_REF}" ] \
  || die "TORCH_VERSION / SAGEATTENTION_REF missing from ${ENV_SRC}"

log "pins from $(basename "${ENV_SRC}")"
printf '   torch         %s (cu130)\n   SageAttention %s\n' \
  "${TORCH_VERSION}" "${SAGEATTENTION_REF}"
printf '   venv          %s\n' "${VENV}"

# --- Host preconditions ------------------------------------------------------
# Fail here rather than 20 minutes into a build. The arch list is what lets
# SageAttention's nvcc pass target Blackwell, so a non-Blackwell GPU would
# compile kernels it cannot run.
log "checking host"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- is this a GPU host?"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader \
  | sed 's/^/   GPU: /'

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
[ "${CC%%.*}" -ge 12 ] 2>/dev/null \
  || die "compute capability ${CC} is not Blackwell (>= 12.0); NVFP4 has no hardware path here"
export TORCH_CUDA_ARCH_LIST="${CC}"

# SageAttention is a torch C++/CUDA extension, and torch.utils.cpp_extension
# refuses to build one whose nvcc major differs from the CUDA that torch itself
# was compiled against -- so a cu130 torch needs a 13.x *toolkit*, not merely a
# 13.x-capable driver. Hosts routinely ship an older toolkit under a new driver
# (a RunPod CUDA 12.8 template on a 580 driver, for one), and /usr/local/cuda
# keeps pointing at it even after a newer one is installed alongside. So search
# for a matching major instead of trusting the symlink or $PATH -- nvcc is often
# not on $PATH at all in a non-login shell.
CUDA_MAJOR=13
nvcc_release() { "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p'; }

NVCC=""
for cand in \
    ${CUDA_HOME:+"${CUDA_HOME}/bin/nvcc"} \
    /usr/local/cuda-${CUDA_MAJOR}.*/bin/nvcc \
    /usr/local/cuda-${CUDA_MAJOR}/bin/nvcc \
    /usr/local/cuda/bin/nvcc \
    "$(command -v nvcc 2>/dev/null || true)"; do
  [ -x "${cand}" ] || continue
  v="$(nvcc_release "${cand}")"
  if [ "${v%%.*}" = "${CUDA_MAJOR}" ]; then NVCC="${cand}"; NVCC_VER="${v}"; break; fi
done

[ -n "${NVCC}" ] || die "no CUDA ${CUDA_MAJOR}.x toolkit found.
       torch is pinned to a cu130 build and cpp_extension will refuse to compile
       SageAttention against a different major, so a 13.x-capable driver is not
       enough on its own.
       Present: $(ls -d /usr/local/cuda-* 2>/dev/null | tr '\n' ' ')
       With NVIDIA's apt repo: apt-get install -y cuda-toolkit-13-0"

# cpp_extension reads CUDA_HOME, so pin it rather than letting it guess.
CUDA_HOME="$(dirname "$(dirname "${NVCC}")")"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
printf '   nvcc: %s (%s)\n' "${NVCC_VER}" "${CUDA_HOME}"

# --- System packages ---------------------------------------------------------
# ffmpeg for video muxing, libgl1/libglib for opencv, python3.12-venv because
# Ubuntu ships venv separately from the interpreter.
log "installing system packages"
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
export DEBIAN_FRONTEND=noninteractive
${SUDO} apt-get update -qq
${SUDO} apt-get install -y --no-install-recommends \
  python3.12 python3.12-venv python3.12-dev \
  git curl ca-certificates ffmpeg libgl1 libglib2.0-0

# --- Virtualenv --------------------------------------------------------------
log "creating venv at ${VENV}"
[ -d "${VENV}" ] || python3.12 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_RETRIES=5 PIP_TIMEOUT=60
pip install --quiet --upgrade pip wheel setuptools

# --- PyTorch -----------------------------------------------------------------
# cu130 index with torch pinned; torchvision/torchaudio are left to pip so the
# pairing is resolved rather than guessed.
log "installing torch ${TORCH_VERSION} (cu130)"
pip install --index-url https://download.pytorch.org/whl/cu130 \
  "torch==${TORCH_VERSION}" torchvision torchaudio

# --- Sources -----------------------------------------------------------------
# Fetching them is the `checkout` make target's job, shared with the onprem path
# so both agree on what lands on disk. Only check that it has happened.
[ -f "${REPO_ROOT}/ComfyUI/requirements.txt" ] \
  || die "no ComfyUI checkout at ${REPO_ROOT}/ComfyUI -- run \`make checkout\` first"
[ -d "${REPO_ROOT}/data/custom_nodes/ComfyUI-KJNodes" ] \
  || die "no KJNodes checkout under data/custom_nodes -- run \`make checkout\` first"

# --- Python dependencies -----------------------------------------------------
# ComfyUI leaves torch/torchvision/torchaudio unpinned, and they are already
# satisfied above, so the cu130 build survives this step.
log "installing ComfyUI requirements"
pip install -r "${REPO_ROOT}/ComfyUI/requirements.txt"

log "installing KJNodes dependencies"
pip install color-matcher matplotlib mss opencv-python-headless "pillow>=10.3.0" ninja packaging

# --- SageAttention, from source ----------------------------------------------
# The PyPI 1.0.6 release predates the int8/fp8 CUDA kernels that the KJNodes
# "MiniMax H3 Memory Efficient Sage Attention Patch" imports -- with 1.0.6 the
# node registers but fails at execution time.
log "building SageAttention @ ${SAGEATTENTION_REF} for sm_${TORCH_CUDA_ARCH_LIST/./}"
SRC="$(mktemp -d)"
trap 'rm -rf "${SRC}"' EXIT
curl -fsSL "https://github.com/thu-ml/SageAttention/archive/${SAGEATTENTION_REF}.tar.gz" \
  | tar -xz -C "${SRC}" --strip-components=1
EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" \
  pip install --no-build-isolation "${SRC}"

# --- Hugging Face CLI --------------------------------------------------------
log "installing huggingface CLI"
pip install "huggingface_hub[cli,hf_transfer]"

# --- Done --------------------------------------------------------------------
cat <<EOF

>> done.

   venv     ${VENV}
   ComfyUI  ${REPO_ROOT}/ComfyUI
   data     ${REPO_ROOT}/data

Next:

  make models    # the 42.5GB weights
  make doctor    # verify the stack
  make up        # start ComfyUI
EOF
