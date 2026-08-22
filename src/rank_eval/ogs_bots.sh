#!/usr/bin/env bash
# Control the three OGS research bots (base / v4 / v5c) as one unit.
#
#   bash src/rank_eval/ogs_bots.sh start     # vLLM server + 3 gtp2ogs in tmux session "ogs"
#   bash src/rank_eval/ogs_bots.sh status    # server, bots, active games, GPU
#   bash src/rank_eval/ogs_bots.sh drain     # stop accepting challenges, finish running games
#   bash src/rank_eval/ogs_bots.sh undrain   # accept challenges again
#   bash src/rank_eval/ogs_bots.sh stop      # kill bots + free the GPU (games in progress time out!)
#   bash src/rank_eval/ogs_bots.sh render    # regenerate config/ogs/bot-*.json5 from the template
#   bash src/rank_eval/ogs_bots.sh ratings   # current 9x9 ratings/ranks of the three bots
#
# Polite pause for a co-tenant who needs the GPU:  drain  → wait until
# `status` shows 0 active games → stop.   Resume later with: start.
# Emergency:  stop  (the ~1 running game per bot is forfeited on time).
#
# Secrets: config/ogs/keys.env (gitignored; see keys.env.example).
set -euo pipefail

REPO=${REPO:-/home/tyliu/katago_workspace}
CFG_DIR="$REPO/config/ogs"
KEYS="$CFG_DIR/keys.env"
SESSION=${SESSION:-ogs}
SERVE="$REPO/src/rank_eval/serve_llm.sh"
BOTS=(base v4 v5c)
declare -A MODEL=([base]=qwen3-8b-base [v4]=go-v4 [v5c]=go-v5c)
declare -A ENGINE=([base]="Qwen3-8B base (no fine-tuning), sampled T=0.6"
                   [v4]="Qwen3-8B GRPO-v4 LoRA checkpoint-5000, sampled T=0.6"
                   [v5c]="Qwen3-8B GRPO-v5c LoRA checkpoint-5000, sampled T=0.6")

cmd=${1:-status}

load_keys() {
  if [[ ! -f "$KEYS" ]]; then
    echo "missing $KEYS — copy keys.env.example and fill in the API keys"; exit 1
  fi
  # shellcheck disable=SC1090
  source "$KEYS"
}

key_of() { local v="BOT_$(echo "$1" | tr a-z A-Z)_APIKEY"; echo "${!v:-}"; }
id_of()  { local v="BOT_$(echo "$1" | tr a-z A-Z)_ID"; echo "${!v:-}"; }
name_of(){ local v="BOT_$(echo "$1" | tr a-z A-Z)_USERNAME"; echo "${!v:-}"; }

render() {
  local decline=${1:-false}
  for b in "${BOTS[@]}"; do
    sed -e "s#__REPO__#$REPO#g" -e "s#__HOME__#$HOME#g" -e "s#__MODEL__#${MODEL[$b]}#g" \
        -e "s#__LABEL__#ogs-$b#g" -e "s#__ENGINE__#${ENGINE[$b]}#g" -e "s#__DECLINE__#$decline#g" \
        "$CFG_DIR/bot-common.json5" > "$CFG_DIR/bot-$b.json5"
  done
  echo "rendered ${#BOTS[@]} configs (decline_new_challenges=$decline)"
}

set_decline() {  # gtp2ogs watches its config file and reloads it live
  for b in "${BOTS[@]}"; do
    sed -i "s#decline_new_challenges: \(true\|false\)#decline_new_challenges: $1#" "$CFG_DIR/bot-$b.json5"
  done
}

start() {
  load_keys
  mkdir -p "$REPO/src/rank_eval/logs/ogs"
  render false
  bash "$SERVE" start
  bash "$SERVE" wait
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n control "echo 'ogs bots control window'; bash"
  fi
  for b in "${BOTS[@]}"; do
    local key; key=$(key_of "$b")
    if [[ -z "$key" ]]; then echo "no API key for $b in keys.env — skipping"; continue; fi
    if tmux list-windows -t "$SESSION" -F '#W' | grep -qx "$b"; then
      echo "bot $b already has a tmux window"; continue
    fi
    tmux new-window -t "$SESSION" -n "$b" \
      "cd $REPO && gtp2ogs -c $CFG_DIR/bot-$b.json5 --apikey $key 2>&1 | tee -a $REPO/src/rank_eval/logs/ogs/console-$b.log; echo '[gtp2ogs $b exited]'; sleep 86400"
    echo "started bot $b (tmux window $SESSION:$b)"
  done
  echo "attach with: tmux attach -t $SESSION   (detach: Ctrl-B then D)"
}

stop() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION" && echo "killed tmux session $SESSION (all gtp2ogs)"
  fi
  pkill -f "src/rank_eval/llm_gtp.py" 2>/dev/null || true
  bash "$SERVE" stop
}

active_games() {  # prints "<n>" active games for a bot id, or "?" if unknown
  local id=$1
  [[ -z "$id" ]] && { echo "?"; return; }
  curl -sf "https://online-go.com/api/v1/players/$id/games/?ended__isnull=true&page_size=1" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin).get("count","?"))' 2>/dev/null || echo "?"
}

status() {
  bash "$SERVE" status
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' windows: $(tmux list-windows -t "$SESSION" -F '#W' | tr '\n' ' ')"
  else
    echo "tmux session '$SESSION': not running"
  fi
  if [[ -f "$KEYS" ]]; then
    load_keys
    for b in "${BOTS[@]}"; do
      local id; id=$(id_of "$b")
      local decl="?"
      [[ -f "$CFG_DIR/bot-$b.json5" ]] && decl=$(grep -o 'decline_new_challenges: [a-z]*' "$CFG_DIR/bot-$b.json5" | awk '{print $2}')
      printf "  bot %-5s user=%-20s id=%-8s active_games=%s decline_new=%s\n" "$b" "$(name_of "$b")" "${id:-?}" "$(active_games "$id")" "$decl"
    done
  fi
}

ratings() {
  load_keys
  local ids=()
  for b in "${BOTS[@]}"; do ids+=("$b=$(id_of "$b")"); done
  python3 "$REPO/src/rank_eval/ogs_rating.py" "${ids[@]}"
}

case "$cmd" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  drain) set_decline true; echo "bots now decline new challenges; wait for active_games=0 in 'status', then 'stop'" ;;
  undrain) set_decline false; echo "bots accept challenges again" ;;
  render) render "${2:-false}" ;;
  ratings) ratings ;;
  *) echo "usage: $0 start|stop|status|drain|undrain|render|ratings"; exit 1 ;;
esac
