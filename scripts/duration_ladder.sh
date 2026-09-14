#!/usr/bin/env bash
#
# Duration ladder: generate the same clip at increasing lengths and record, per
# rung, the server-side execution time, the nvidia-smi VRAM peak and what
# actually came out. One variable -- length -- everything else pinned.
#
# Why: "how long a clip can this machine make" has two different answers.
# Where VRAM runs out moves with the hardware. Where the model stops producing
# a picture moved with nothing when measured on 96 GB -- 1,467 frames (61 s)
# was a film, 2,147 frames (89 s) a flat grey field from the first frame, with
# 30 GB still free. A ladder finds both, in order, and stops at the first rung
# that fails, because every longer rung would fail the same way and each one
# costs an hour.
#
# Run it on the machine that owns the GPU: the VRAM poll is local nvidia-smi.
# Point SERVER at a ComfyUI on this host; Docker or venv, either is fine, since
# everything is read through the HTTP API rather than the server's log.
#
# Usage:
#   scripts/duration_ladder.sh [DURATION...]              # default: 30 61 89 120 149
#   SEED=43 scripts/duration_ladder.sh 5 30
#   TEMPLATE=templates/x.json scripts/duration_ladder.sh   # e.g. a turbo variant
#
# Environment:
#   SERVER     ComfyUI base URL         default http://localhost:${COMFY_PORT} (from .env)
#   SEED       held fixed across rungs  default 43 -- the seed of the 2026-08-11 A/B
#   TEMPLATE   generate.py --template   default: generate.py's own default (accel t2v)
#   PROMPT     passed verbatim          default: the prompt embedded in TEMPLATE
#   GPU_INDEX  nvidia-smi -i            default 0
#   OUT        results directory       default data/ladder/<UTC timestamp>
#
# Output, under $OUT:
#   results.tsv         one row per rung (duration, frames, clip s, exec s, VRAM peak, file, prompt_id)
#   vram_<DUR>.txt      raw 2 s nvidia-smi samples -- the first minutes are a transient
#                       (text encoder + DiT + allocator overshoot); steady-state sampling
#                       sits well below the peak, so read the trace, not just the max
#   generate_<DUR>.log  generate.py's output for that rung
#   conditions.txt      server, seed, template, GPU, prompt hash
#   prompt.txt          the prompt, verbatim
#
# Durations go through generate.py, so the 17k+5 frame grid is enforced on the
# server. Note that DURATION=150 rounds up to 3,609 frames and is refused by the
# node's max of 3,600; 149 (3,592 frames) is the practical ceiling.

set -uo pipefail   # not -e: a failing rung is a result, and is handled below

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- Conditions --------------------------------------------------------------
ENV_SRC=.env; [ -f "${ENV_SRC}" ] || ENV_SRC=.env.example
COMFY_PORT="${COMFY_PORT:-$(grep -E '^COMFY_PORT=' "${ENV_SRC}" 2>/dev/null | cut -d= -f2)}"
SERVER="${SERVER:-http://localhost:${COMFY_PORT:-8188}}"
SEED="${SEED:-43}"
GPU_INDEX="${GPU_INDEX:-0}"
OUT="${OUT:-data/ladder/$(date -u +%Y%m%dT%H%M%SZ)}"
RUNGS=("$@"); [ "${#RUNGS[@]}" -gt 0 ] || RUNGS=(30 61 89 120 149)

# Default to generate.py's own default template, read from generate.py, so the
# two cannot drift apart.
TEMPLATE="${TEMPLATE:-$(python3 -c 'import sys; sys.path.insert(0, "scripts"); import generate; print(generate.DEFAULT_TEMPLATE_T2V)')}"
[ -f "${TEMPLATE}" ] || die "template not found: ${TEMPLATE}"

# The template's own prompt, unless one was given. Same text on every rung is
# what makes the rungs comparable.
if [ -z "${PROMPT:-}" ]; then
  PROMPT="$(python3 - "${TEMPLATE}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for v in d.values():
    if v.get("class_type") == "MiniMaxH3ImageToVideo":
        print(v["inputs"]["prompt"]); break
PY
)"
fi
[ -n "${PROMPT}" ] || die "no prompt: none in ${TEMPLATE} and PROMPT not set"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- run this on the GPU host"
curl -fsS --max-time 5 "${SERVER}/system_stats" >/dev/null \
  || die "ComfyUI is not answering at ${SERVER} (make up?)"
if command -v ffprobe >/dev/null; then HAVE_FFPROBE=1; else
  HAVE_FFPROBE=0; echo "note: ffprobe not on this host; frames/clip_s columns will be '-'" >&2
fi

mkdir -p "${OUT}"
RESULTS="${OUT}/results.tsv"
printf 'duration_s\tframes\tclip_s\texec_s\tvram_peak_MiB\toutput\tprompt_id\n' > "${RESULTS}"
{
  echo "server=${SERVER}"
  echo "seed=${SEED}"
  echo "template=${TEMPLATE}"
  echo "gpu=$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=name,memory.total --format=csv,noheader)"
  echo "prompt_sha256=$(printf '%s' "${PROMPT}" | sha256sum | cut -c1-16)"
  echo "rungs=${RUNGS[*]}"
} > "${OUT}/conditions.txt"
printf '%s' "${PROMPT}" > "${OUT}/prompt.txt"

# Server-side execution time from /history: the span between the
# execution_start and execution_success messages, which ComfyUI stamps in ms.
# Reading it here rather than grepping the server log keeps the measurement
# identical under Docker (docker logs) and a venv (a file), and sidesteps the
# log's switch to HH:MM:SS past ten minutes.
exec_seconds() {
  python3 - "${SERVER}" "$1" <<'PY'
import json, sys, urllib.request
server, pid = sys.argv[1], sys.argv[2]
try:
    h = json.load(urllib.request.urlopen(f"{server}/history/{pid}", timeout=30))
    t = {m[0]: m[1]["timestamp"] for m in h[pid]["status"]["messages"]
         if m[0] in ("execution_start", "execution_success")}
    print(f"{(t['execution_success'] - t['execution_start']) / 1000:.2f}")
except Exception:
    print("-")
PY
}

# --- The ladder --------------------------------------------------------------
echo ">> ladder: ${RUNGS[*]}  seed=${SEED}  server=${SERVER}"
echo ">> results: ${RESULTS}"

for DUR in "${RUNGS[@]}"; do
  echo
  echo "== DURATION=${DUR}  $(date -Is)"

  VRAM="${OUT}/vram_${DUR}.txt"; : > "${VRAM}"
  ( while :; do
      nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used --format=csv,noheader,nounits >> "${VRAM}"
      sleep 2
    done ) &
  POLL=$!

  LOG="${OUT}/generate_${DUR}.log"
  python3 scripts/generate.py --prompt "${PROMPT}" --duration "${DUR}" --seed "${SEED}" \
      --server "${SERVER}" --template "${TEMPLATE}" --timeout 86400 2>&1 | tee "${LOG}"
  rc=${PIPESTATUS[0]}

  kill "${POLL}" 2>/dev/null; wait "${POLL}" 2>/dev/null
  peak="$(sort -n "${VRAM}" | tail -1)"
  pid="$(sed -n 's/^submitted: prompt_id=\([^ ]*\).*/\1/p' "${LOG}" | head -1)"
  out="$(sed -n 's/^output: //p' "${LOG}" | head -1)"

  if [ "${rc}" -ne 0 ] || [ -z "${out}" ]; then
    printf '%s\t-\t-\tFAILED(rc=%s)\t%s\t-\t%s\n' \
      "${DUR}" "${rc}" "${peak:--}" "${pid:--}" >> "${RESULTS}"
    echo "!! DURATION=${DUR} failed (rc=${rc}) -- stopping here: every longer rung would too"
    break
  fi

  exec_s="$(exec_seconds "${pid}")"
  frames=-; clip=-
  if [ "${HAVE_FFPROBE}" -eq 1 ] && [ -f "${out}" ]; then
    frames="$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nw=1:nk=1 "${out}")"
    clip="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "${out}")"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${DUR}" "${frames}" "${clip}" "${exec_s}" "${peak:--}" "$(basename "${out}")" "${pid}" >> "${RESULTS}"
  echo ">> DURATION=${DUR}: ${frames} frames, ${clip} s, exec ${exec_s} s, VRAM peak ${peak:--} MiB -> $(basename "${out}")"
done

echo
echo "== ${RESULTS}"
awk -F'\t' '{ printf "  %-10s %-7s %-10s %-14s %-13s %s\n", $1, $2, $3, $4, $5, $6 }' "${RESULTS}"
echo
echo "Whether a rung is a picture or a grey field is not in the table. Look at the output."
