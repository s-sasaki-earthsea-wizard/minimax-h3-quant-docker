#!/usr/bin/env bash
#
# Run a generation command against a ComfyUI on another machine, over an SSH
# tunnel that lives exactly as long as the command.
#
# Why a tunnel rather than exposing the port: ComfyUI has no authentication of
# any kind -- cli_args.py offers --tls-keyfile/--tls-certfile and nothing else --
# so a reachable port is an open invitation to submit jobs and read and write
# files on that host. SSH is the only thing authenticating this.
#
# Only the HTTP API travels. The prompt-writing LLM stays on the machine you run
# this from, which is the point: Ollama needs neither a second copy of a 15 GB
# model on the remote host nor to contend for VRAM with a generation already in
# flight there.
#
# Usage: with_remote.sh <ssh-host> <local-port> <remote-port> <command...>
#
#   ssh-host     a Host from ~/.ssh/config, or user@host
#   local-port   forwarded here; must be free
#   remote-port  where ComfyUI listens over there
#
set -euo pipefail

[ $# -ge 4 ] || { echo "usage: $0 <ssh-host> <local-port> <remote-port> <command...>" >&2; exit 1; }
HOST="$1"; LOCAL_PORT="$2"; REMOTE_PORT="$3"; shift 3

SOCK="$(mktemp -u "${TMPDIR:-/tmp}/comfy-tunnel-XXXXXXXX")"

# A control socket makes teardown exact: -O exit closes the one connection this
# script opened, rather than pkill'ing a pattern that could match someone else's
# tunnel -- or, as happens with `pkill -f`, the very shell running the pattern.
cleanup() { ssh -S "${SOCK}" -O exit "${HOST}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo ">> tunnel: localhost:${LOCAL_PORT} -> ${HOST}:${REMOTE_PORT}" >&2
if ! ssh -f -N -M -S "${SOCK}" -o ExitOnForwardFailure=yes \
        -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${HOST}"; then
  echo "error: could not open the tunnel. Is ${HOST} in ~/.ssh/config, and is" >&2
  echo "       local port ${LOCAL_PORT} free? (override with REMOTE_PORT=)" >&2
  exit 1
fi

# ExitOnForwardFailure only proves the forward was set up, not that anything is
# listening at the far end, so confirm before spending an LLM call on a prompt.
for _ in $(seq 30); do
  curl -fsS --max-time 2 "http://localhost:${LOCAL_PORT}/system_stats" >/dev/null 2>&1 && break
  sleep 1
done
if ! curl -fsS --max-time 2 "http://localhost:${LOCAL_PORT}/system_stats" >/dev/null 2>&1; then
  echo "error: tunnel is up but nothing answers on ${HOST}:${REMOTE_PORT}." >&2
  echo "       Is ComfyUI running there? (make status, over there)" >&2
  exit 1
fi

# `set -e` would take the exit before rc could be read, so capture it in the
# one form that is exempt.
rc=0
"$@" || rc=$?

# Outputs are written on the far side; say so, because the path generate.py
# prints is relative and reads as though it were local.
if [ "${rc}" -eq 0 ]; then
  cat >&2 <<EOF

>> any output landed on ${HOST}, not here. To fetch it:
   rsync -avP ${HOST}:<repo>/data/output/video/ ./pod-videos/
EOF
fi
exit "${rc}"
