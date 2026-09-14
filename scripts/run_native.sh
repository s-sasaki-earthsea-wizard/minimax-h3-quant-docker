#!/usr/bin/env bash
#
# Start / stop ComfyUI from the venv built by scripts/setup_native.sh.
#
# Detached on purpose: a 3592-frame generation runs for hours, far longer than an
# SSH session can be relied on to stay up, so the server is started under nohup
# with its streams redirected rather than left attached to the login shell.
#
# Flag rationale, and how it differs from docker/compose.yaml:
#
#   --base-directory      models, custom_nodes, input, output, user, temp.
#   --database-url        set explicitly because it does NOT follow
#                         --base-directory: cli_args.py derives it from its own
#                         __file__, landing on <checkout>/user/comfyui.db, a
#                         directory the git checkout does not contain. Left
#                         alone it fails with "unable to open database file".
#   NOT --use-sage-attention
#                         a code-path omission in the MiniMax implementation can
#                         produce noise output when SageAttention is forced
#                         globally. Use the KJNodes "MiniMax H3 Memory Efficient
#                         Sage Attention Patch" node inside the workflow.
#   NOT --disable-pinned-memory, NOT --fast-disk
#                         both are 16 GB tuning: they trade speed for headroom by
#                         keeping weights off the card. On a host with enough VRAM
#                         to hold the 21 GB DiT resident they cost time and buy
#                         nothing. compose.yaml keeps them; this does not.
#
# Usage: scripts/run_native.sh {start|stop|status|logs}
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$(dirname "${REPO_ROOT}")/venv}"
DATA="${REPO_ROOT}/data"
LOG="${DATA}/comfyui.log"
PIDFILE="${DATA}/comfyui.pid"

ENV_SRC="${REPO_ROOT}/.env"
[ -f "${ENV_SRC}" ] || ENV_SRC="${REPO_ROOT}/.env.example"
COMFY_PORT="$(grep -E '^COMFY_PORT=' "${ENV_SRC}" 2>/dev/null | head -1 | cut -d= -f2)"
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_EXTRA_ARGS="$(grep -E '^COMFY_EXTRA_ARGS=' "${ENV_SRC}" 2>/dev/null | head -1 | cut -d= -f2-)"

running() {
  [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null
}

case "${1:-}" in
  start)
    if running; then
      echo "already running (pid $(cat "${PIDFILE}"))"; exit 0
    fi
    [ -x "${VENV}/bin/python" ] || { echo "no venv at ${VENV} -- run scripts/setup_native.sh" >&2; exit 1; }
    mkdir -p "${DATA}/user"

    # Blackwell sm_120 for any Triton/JIT kernel compiled at runtime, and keep
    # the HF cache on the same volume as the weights.
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
    export HF_HOME="${HF_HOME:-${DATA}/.cache/huggingface}"
    export HF_XET_HIGH_PERFORMANCE=1

    # nohup rather than setsid: SIGHUP immunity is what surviving a dropped SSH
    # session actually needs, and setsid only forks when it is not already a
    # process group leader -- so $! is sometimes setsid's own pid and sometimes
    # the server's, depending on how this script was invoked. A pidfile that is
    # right by accident is worse than no pidfile.
    # shellcheck disable=SC2086
    nohup "${VENV}/bin/python" "${REPO_ROOT}/ComfyUI/main.py" \
      --listen 0.0.0.0 \
      --port "${COMFY_PORT}" \
      --base-directory "${DATA}" \
      --database-url "sqlite:///${DATA}/user/comfyui.db" \
      ${COMFY_EXTRA_ARGS} \
      >>"${LOG}" 2>&1 < /dev/null &

    echo $! > "${PIDFILE}"
    echo "started (pid $(cat "${PIDFILE}")) -> http://localhost:${COMFY_PORT}"
    echo "logs: ${LOG}"
    ;;

  stop)
    running || { echo "not running"; rm -f "${PIDFILE}"; exit 0; }
    pid="$(cat "${PIDFILE}")"
    # SIGTERM first: a generation in flight should get the chance to release the
    # GPU cleanly rather than leaving 50 GB pinned until the driver notices.
    kill "${pid}"
    for _ in $(seq 30); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
    kill -0 "${pid}" 2>/dev/null && { echo "still up after 30s, sending SIGKILL"; kill -9 "${pid}"; }
    rm -f "${PIDFILE}"
    echo "stopped"
    ;;

  status)
    if running; then
      echo "running (pid $(cat "${PIDFILE}")) on port ${COMFY_PORT}"
      curl -fsS "http://localhost:${COMFY_PORT}/system_stats" >/dev/null \
        && echo "  /system_stats: OK" || echo "  /system_stats: not answering yet"
    else
      echo "not running"
    fi
    ;;

  logs) tail -f "${LOG}" ;;

  *) echo "usage: $0 {start|stop|status|logs}" >&2; exit 1 ;;
esac
