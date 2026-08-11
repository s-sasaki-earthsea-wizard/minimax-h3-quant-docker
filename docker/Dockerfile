# syntax=docker/dockerfile:1.7
#
# ComfyUI runtime for MiniMax-H3 on a single 16 GB Blackwell GPU (RTX 5080, sm_120).
#
# The CUDA 13.0 base and torch cu130 wheels are deliberate: NVFP4 has no hardware
# path before Blackwell, and the quantised Qwen3-VL text encoder shipped by
# Comfy-Org is nvfp4_awq. This mirrors the known-good reference stack
# (PyTorch 2.12.0+cu130) instead of the host's torch 2.10+cu128.
#
# ComfyUI itself is NOT baked into the image -- it is bind-mounted from the host
# so it stays inspectable and updatable with plain git. Only its dependencies
# live here.

FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ARG COMFYUI_REF=v0.30.2
ARG TORCH_VERSION=2.12.0
ARG SAGEATTENTION_REF=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
ARG HOST_UID=1000
ARG HOST_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_RETRIES=5 \
    PIP_TIMEOUT=60 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=12.0

# Network resilience. The host resolves archive.ubuntu.com to AAAA records while
# the default Docker bridge has no IPv6 route, and apt's parallel fetches can
# overrun the LAN resolver mid-build -- both surface as
# "Could not resolve 'archive.ubuntu.com'" after a successful `apt-get update`.
RUN printf '%s\n' \
        'Acquire::Retries "5";' \
        'Acquire::ForceIPv4 "true";' \
        'Acquire::http::Timeout "30";' \
        'Acquire::https::Timeout "30";' \
        > /etc/apt/apt.conf.d/99-resilient-fetch

# ffmpeg is required for video muxing, libgl1/libglib for opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        git curl ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip wheel setuptools

# --- PyTorch: cu130 index, pinned torch; let pip resolve the matching
# --- torchvision/torchaudio rather than guessing the pairing.
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
        "torch==${TORCH_VERSION}" torchvision torchaudio

# --- ComfyUI runtime dependencies, pinned to the same ref we check out on the
# --- host. torch/torchvision/torchaudio are unpinned there and already
# --- satisfied above, so the cu130 build survives this step.
ADD https://raw.githubusercontent.com/Comfy-Org/ComfyUI/${COMFYUI_REF}/requirements.txt \
    /tmp/comfyui-requirements.txt
RUN pip install -r /tmp/comfyui-requirements.txt

# --- ComfyUI-KJNodes dependencies, plus build helpers for the SageAttention
# --- compile below (packaging is imported by its setup.py, ninja parallelises
# --- the extension build).
RUN pip install \
        color-matcher matplotlib mss opencv-python-headless "pillow>=10.3.0" \
        ninja packaging

# --- SageAttention, built from source. The PyPI 1.0.6 release predates the
# --- int8/fp8 CUDA kernels (per_warp_int8_cuda, per_channel_fp8, ...) that
# --- KJNodes' "MiniMax H3 Memory Efficient Sage Attention Patch" imports --
# --- with 1.0.6 the node registers but fails at execution time.
# --- TORCH_CUDA_ARCH_LIST=12.0 lets setup.py target sm_120 without a GPU
# --- visible at build time, and keeps the nvcc pass Blackwell-only.
ADD https://github.com/thu-ml/SageAttention/archive/${SAGEATTENTION_REF}.tar.gz \
    /tmp/sageattention.tar.gz
RUN mkdir /tmp/sageattention \
    && tar -xzf /tmp/sageattention.tar.gz -C /tmp/sageattention --strip-components=1 \
    && EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" \
       pip install --no-build-isolation /tmp/sageattention \
    && rm -rf /tmp/sageattention /tmp/sageattention.tar.gz

# --- Hugging Face CLI, used by scripts/download_models.sh inside the container
# --- so model downloads do not depend on the host Python environment.
RUN pip install "huggingface_hub[cli,hf_transfer]"

# --- Match the host user so bind-mounted files stay writable and unowned by root.
RUN userdel -r ubuntu 2>/dev/null || true; \
    groupadd -g "${HOST_GID}" comfy && \
    useradd -u "${HOST_UID}" -g "${HOST_GID}" -m -s /bin/bash comfy

USER comfy
WORKDIR /opt/ComfyUI
EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--base-directory", "/data"]
