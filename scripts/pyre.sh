#!/usr/bin/env bash
# Unix counterpart of pyre.ps1. Starts the pyre binary directly (no PowerShell, no Python).
#   ./scripts/pyre.sh -m /path/to/Qwen3.8-Flash-Next-IQ3_S
#   PYRE_QN_QUANT=IQ3_S ./scripts/pyre.sh -m /path/to/snapshot-with-several-quants
#   ./scripts/pyre.sh --stop
#   ./scripts/pyre.sh --dry-run -m /path/to/model
# The Windows launcher (pyre.ps1) is the config-file path. This script only wraps `pyre serve`.
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
port="${PYRE_PORT:-8099}"
host="${PYRE_HOST:-127.0.0.1}"
backend="${PYRE_BACKEND:-cuda}"
model="${PYRE_MODEL:-}"
quant="${PYRE_QN_QUANT:-}"
watch=0
dry=0
stop=0
foreground=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) model=${2:-}; shift 2 ;;
    -q|--quant) quant=${2:-}; shift 2 ;;
    -p|--port) port=${2:-}; shift 2 ;;
    -b|--backend) backend=${2:-}; shift 2 ;;
    --host) host=${2:-}; shift 2 ;;
    --watch) watch=1; shift ;;
    --dry-run) dry=1; shift ;;
    --stop) stop=1; shift ;;
    --background) foreground=0; shift ;;
    -h|--help)
      echo "usage: ./scripts/pyre.sh -m MODEL_DIR [--quant IQ3_S] [--port 8099] [--watch] [--stop] [--dry-run]"
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

state=${XDG_STATE_HOME:-$HOME/.local/state}/pyre
pidfile=$state/pyre-server.pid
mkdir -p "$state"

find_exe() {
  local c
  for c in \
    "$root/target/release/pyre" \
    "$root/target/debug/pyre" \
    "$root/target/x86_64-pc-windows-msvc/release/pyre.exe" \
    "$root/target/release/pyre.exe"
  do
    if [[ -x $c || -f $c ]]; then printf '%s\n' "$c"; return 0; fi
  done
  if command -v pyre >/dev/null 2>&1; then command -v pyre; return 0; fi
  return 1
}

if [[ $stop -eq 1 ]]; then
  if [[ -f $pidfile ]]; then
    pid=$(cat "$pidfile" || true)
    if [[ -n ${pid:-} ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      echo "sent SIGTERM to $pid"
    else
      echo "not running"
    fi
    rm -f "$pidfile"
  else
    echo "not running"
  fi
  exit 0
fi

if [[ -z $model ]]; then
  echo "pass -m with a model directory, a .gguf, a store name, or an HF repo id" >&2
  exit 2
fi

exe=$(find_exe || true)
args=(serve --host "$host" --port "$port" -b "$backend" -m "$model")
if [[ -n $quant ]]; then export PYRE_QN_QUANT=$quant; fi

if [[ $dry -eq 1 ]]; then
  echo "exe  : ${exe:-not found (cargo build --release)}"
  echo "args : ${args[*]}"
  echo "env  : PYRE_QN_QUANT=${PYRE_QN_QUANT:-}"
  echo "url  : http://127.0.0.1:$port"
  exit 0
fi
if [[ -z ${exe:-} ]]; then
  echo "pyre executable not found. From the repo: cargo build --release" >&2
  exit 1
fi

start_one() {
  if [[ $foreground -eq 1 && $watch -eq 0 ]]; then
    exec "$exe" "${args[@]}"
  fi
  if [[ $foreground -eq 0 ]]; then
    nohup "$exe" "${args[@]}" >>"$state/pyre.log" 2>&1 &
    echo $! >"$pidfile"
    echo "logs: $state/pyre.log"
    return
  fi
  "$exe" "${args[@]}" &
  echo $! >"$pidfile"
  wait $! || true
}

echo "pyre: http://127.0.0.1:$port"
echo "model: $model"
echo "stop: ./scripts/pyre.sh --stop"
if [[ $watch -eq 0 ]]; then
  start_one
  exit 0
fi
while true; do
  t0=$(date +%s)
  set +e
  "$exe" "${args[@]}" &
  echo $! >"$pidfile"
  wait $! 
  code=$?
  set -e
  rm -f "$pidfile"
  if [[ $code -eq 87 || $code -eq 143 || $code -eq 130 ]]; then
    echo "pyre exited $code; not restarting"
    exit 0
  fi
  now=$(date +%s)
  if [[ $((now - t0)) -lt 5 ]]; then
    echo "pyre exited $code in under 5s; not restarting" >&2
    exit "$code"
  fi
  echo "pyre exited $code; restarting in 2s"
  sleep 2
done
