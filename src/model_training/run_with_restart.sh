#!/usr/bin/env bash
# Auto-restarting wrapper for the v5 GRPO training run.
#
# Why this exists: exp05c died at step 1616 with
#   Fatal Python error: none_dealloc: deallocating None:
#   bug likely caused by a refcount error in a C extension
# a random refcount bug in one of the container's C extensions (vLLM,
# FlashInfer, Triton, numba are all loaded). It is unrelated to the training
# config and cannot be fixed from here; at that rate a 10,000-step run will hit
# it several more times. This wrapper restarts from the newest checkpoint
# whenever the trainer dies abnormally, and gets out of the way when it doesn't.
#
# Usage — from the repo root inside the training container, under tmux:
#   bash src/model_training/run_with_restart.sh
#
# Keep one continuous W&B curve across restarts by exporting the run id:
#   WANDB_RUN_ID=lc4hyp23 WANDB_RESUME=allow \
#     bash src/model_training/run_with_restart.sh
#
# Knobs (environment):
#   MAX_RESTARTS          give up after this many restarts       (default 20)
#   MIN_HEALTHY_SECONDS   a run shorter than this counts as a
#                         fast failure; 3 consecutive ones abort (default 600)
#   RESTART_DELAY         seconds to wait before restarting      (default 30)
#
# Exit codes: 0 = training completed; otherwise the last trainer exit code, or
# 2 if the wrapper gave up (restart budget exhausted / crash loop).

set -uo pipefail

TRAIN_SCRIPT="src/model_training/train_grpo_v5.py"
WRAPPER_LOG="src/model_training/logs/run_with_restart.log"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
MIN_HEALTHY_SECONDS="${MIN_HEALTHY_SECONDS:-600}"
RESTART_DELAY="${RESTART_DELAY:-30}"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "[wrapper] $TRAIN_SCRIPT not found — run this from the repo root." >&2
    exit 2
fi

# Single source of truth for the output dir: read it back out of the trainer.
OUTPUT_DIR="$(grep -m1 '^OUTPUT_DIR = ' "$TRAIN_SCRIPT" | cut -d'"' -f2)"
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "[wrapper] could not read OUTPUT_DIR from $TRAIN_SCRIPT" >&2
    exit 2
fi

mkdir -p "$(dirname "$WRAPPER_LOG")"

log() {
    printf '[wrapper %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$WRAPPER_LOG"
}

# Ctrl-C / docker stop must end the whole thing, not trigger a restart.
stop_requested=0
trap 'stop_requested=1' INT TERM

# Resume only when there is something to resume from, so a fresh run (or a
# crash before the first checkpoint) starts correctly instead of erroring out.
resume_flag() {
    if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null 2>&1; then
        echo auto
    else
        echo ""
    fi
}

log "=== wrapper start === output_dir=${OUTPUT_DIR} max_restarts=${MAX_RESTARTS}"

restarts=0
fast_failures=0

while :; do
    resume="$(resume_flag)"
    if [[ -n "$resume" ]]; then
        latest="$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null \
            | awk -F'checkpoint-' '{print $2, $0}' | sort -n | tail -1 | cut -d' ' -f2-)"
        log "attempt $((restarts + 1)): resuming from ${latest}"
    else
        log "attempt $((restarts + 1)): fresh start (no checkpoint in ${OUTPUT_DIR})"
    fi

    started_at=$(date +%s)
    RESUME_FROM="$resume" python "$TRAIN_SCRIPT"
    code=$?
    elapsed=$(( $(date +%s) - started_at ))

    if [[ "$stop_requested" -eq 1 ]]; then
        log "interrupted by user after ${elapsed}s (exit ${code}) — not restarting"
        exit "$code"
    fi

    if [[ "$code" -eq 0 ]]; then
        log "training finished normally after ${elapsed}s"
        exit 0
    fi

    log "trainer died with exit ${code} after ${elapsed}s"

    # A run that dies almost immediately is a broken config/environment, not the
    # random C-extension crash — restarting it forever would only hide the error.
    if [[ "$elapsed" -lt "$MIN_HEALTHY_SECONDS" ]]; then
        fast_failures=$((fast_failures + 1))
        log "fast failure ${fast_failures}/3 (ran <${MIN_HEALTHY_SECONDS}s)"
        if [[ "$fast_failures" -ge 3 ]]; then
            log "GIVING UP: 3 consecutive fast failures — this is not the random"
            log "crash. Read the traceback above before restarting."
            exit 2
        fi
    else
        fast_failures=0
    fi

    restarts=$((restarts + 1))
    if [[ "$restarts" -gt "$MAX_RESTARTS" ]]; then
        log "GIVING UP: restart budget (${MAX_RESTARTS}) exhausted"
        exit 2
    fi

    log "restarting in ${RESTART_DELAY}s..."
    sleep "$RESTART_DELAY"
    if [[ "$stop_requested" -eq 1 ]]; then
        log "interrupted during restart delay — exiting"
        exit "$code"
    fi
done
