#!/usr/bin/env bash
#
# Download the MiniMax-H3 weights sized for a 16 GB GPU.
#
# Designed to run INSIDE the container (`make models`), where the Hugging Face
# CLI is guaranteed present. MODELS_DIR defaults to the container path; override
# it to run on the host.
#
# Repository is public (gated=false), so no HF token is required.
#
# Variant notes:
#   pruned       - AdaLN branches stripped. The MiniMax README states ~13B of the
#                  33B parameters live in AdaLN branches whose modulation outputs
#                  can be precomputed, so they are not needed for inference-only
#                  use. 66.28 GB - 40.23 GB = 26 GB = 13B params at bf16.
#   int8_convrot - ComfyUI's official template default.
#   fp8_scaled   - Same size, fp8 tensor-core path. Interchangeable in practice.
#   nvfp4_awq    - Text encoder. NVFP4 has no hardware path before Blackwell,
#                  which is exactly what an RTX 5080 provides.

set -euo pipefail

REPO="Comfy-Org/MiniMax-H3"
MODELS_DIR="${MODELS_DIR:-/data/models}"
DIT_VARIANT="${DIT_VARIANT:-int8_convrot}"   # int8_convrot | fp8_scaled
TASKS="${TASKS:-fl2va}"                      # space separated: fl2va ref2va

files=(
  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "vae/minimax_h3_video_vae_fp16.safetensors"
  "vae/minimax_h3_audio_vae_fp32.safetensors"
)

for task in $TASKS; do
  files+=("diffusion_models/minimax_h3_${task}_pruned_${DIT_VARIANT}.safetensors")
done

echo "Repository : ${REPO}"
echo "Destination: ${MODELS_DIR}"
echo "Tasks      : ${TASKS} (pruned/${DIT_VARIANT})"
echo
echo "Requesting ${#files[@]} files:"
printf '  %s\n' "${files[@]}"
echo

mkdir -p "${MODELS_DIR}"

# `hf download` is a Typer CLI with the signature:
#     hf download [OPTIONS] REPO_ID [FILENAMES]...
# --include takes ONE glob per occurrence, so passing a list to a single
# --include silently routes the remainder into FILENAMES and prints
# "Ignoring --include since filenames have been explicitly set" -- which drops
# files without a non-zero exit. Pass them positionally instead.
#
# The repository layout already matches the diffusion_models/ text_encoders/
# vae/ names ComfyUI expects under models/. Resumable: just re-run on failure.
hf download "${REPO}" "${files[@]}" --local-dir "${MODELS_DIR}"

# Verify every requested file actually landed. The CLI has more than one way to
# come back successful having fetched a subset, so do not trust its exit code.
echo
missing=0
for f in "${files[@]}"; do
  if [[ -s "${MODELS_DIR}/${f}" ]]; then
    size=$(stat -c %s "${MODELS_DIR}/${f}")
    printf '  %7.2f GB  %s\n' "$(echo "${size}" | awk '{print $1/1e9}')" "${f}"
  else
    printf '  MISSING     %s\n' "${f}"
    missing=$((missing + 1))
  fi
done

if (( missing > 0 )); then
  echo
  echo "ERROR: ${missing} of ${#files[@]} requested file(s) missing. Re-run to resume."
  exit 1
fi

echo
echo "All ${#files[@]} files present."
du -sh "${MODELS_DIR}"
