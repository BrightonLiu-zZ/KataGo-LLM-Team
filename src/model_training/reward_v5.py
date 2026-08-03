"""
v5 reward components — pure Python, no torch/trl imports, so the unit tests
(`test_reward_v5.py`) run on any host without the training environment.

Layers (see ADR 0006):
  1. Gate            — unchanged from v4: -1 no MOVE / -0.5 illegal / 0 legal.
  2. Core            — unchanged from v4: log-policy + scoreLead rank,
                       weights adapted to root_winrate closeness.
  3. Reasoning       — NEW: three rule-based terms that finally give the
                       REASONING text a gradient. Total magnitude <= 0.2 so it
                       can never outweigh the KataGo core signal.
       consistency   +0.05 if the chosen coordinate appears in the reasoning
       length window -0.10 * deficit if reasoning is shorter than 200 chars
       group dedup   -0.10 * excess pairwise bigram overlap vs the other
                     completions of the same prompt (breaks reward ties when
                     all generations pick the same move -> attacks zero-std)
"""
import math
import re
from itertools import groupby
from typing import List, Optional

VALID_COLS = set("ABCDEFGHJ")
VALID_ROWS = set(str(i) for i in range(1, 10))

# --- Reasoning-layer constants (referenced by tests; tune here only) ---
CONSISTENCY_BONUS = 0.05
LENGTH_FULL_CREDIT_CHARS = 200   # ~50 tokens; exp04 collapsed to ~100 chars
LENGTH_MAX_PENALTY = -0.10
DEDUP_JACCARD_THRESHOLD = 0.35   # same-position reasonings legitimately overlap
DEDUP_MAX_PENALTY = -0.10


# ================= v4-inherited utilities =================

def is_valid_9x9_coord(coord: str) -> bool:
    coord = coord.upper().strip()
    if coord == "PASS":
        return True
    if len(coord) < 2:
        return False
    col, row = coord[0], coord[1:]
    return (col in VALID_COLS) and (row in VALID_ROWS)


def extract_move(text: str) -> Optional[str]:
    match = re.search(r"MOVE:\s*([A-HJ-Z][1-9]|PASS)", text, re.IGNORECASE)
    if match:
        move = match.group(1).upper()
        if is_valid_9x9_coord(move):
            return move
    return None


def check_spatial_legality(prompt_text: str, move_str: str) -> float:
    """Returns 0.0 if move is on an empty '.', else a negative penalty."""
    if move_str == "PASS":
        return 0.0

    col_char, row_char = move_str[0], move_str[1]

    lines = prompt_text.split("\n")
    board_lines = []
    for line in lines:
        if re.match(r"^\s*[1-9]\s+([.XO]\s+)*[.XO]", line):
            board_lines.append(line)

    if len(board_lines) != 9:
        return -0.1

    row_idx = 9 - int(row_char)
    col_mapping = {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
        "F": 5, "G": 6, "H": 7, "J": 8,
    }
    col_idx = col_mapping.get(col_char)
    if col_idx is None:
        return -0.5

    row_elements = board_lines[row_idx].strip().split()
    if len(row_elements) >= 10:
        target_spot = row_elements[col_idx + 1]
        if target_spot != ".":
            return -0.8

    return 0.0


def compute_r_policy_log(move: str, evals: dict) -> float:
    """Log-scaled policy reward in [0, 1]."""
    if move == "PASS":
        entry = evals.get("pass") or evals.get("PASS") or {}
    else:
        entry = evals.get(move, {})

    p = entry.get("policy")
    if p is None or p <= 0:
        return 0.0
    return max(0.0, min(1.0, (math.log10(max(p, 1e-6)) + 4.0) / 4.0))


def compute_r_rank(move: str, evals: dict) -> float:
    """Percentile rank by scoreLead among all evaluated legal moves."""
    scored = []
    for coord, metrics in evals.items():
        if not isinstance(metrics, dict):
            continue
        sl = metrics.get("scoreLead")
        pol = metrics.get("policy", -1)
        if sl is not None and pol >= 0:
            scored.append((coord.upper(), float(sl)))

    if not scored:
        return 0.0

    scored.sort(key=lambda x: x[1], reverse=True)
    n = len(scored)

    move_upper = move.upper() if move != "PASS" else "PASS"
    for rank, (coord, _) in enumerate(scored):
        if coord == move_upper:
            return 1.0 - rank / max(n - 1, 1)

    return 0.0


def composite_core(move: str, evals: dict, rwr) -> float:
    """v4 core: adaptive-weighted log-policy + rank. Returns 0.0 for None move."""
    if not move:
        return 0.0
    r_policy = compute_r_policy_log(move, evals)
    r_rank = compute_r_rank(move, evals)
    rwr_val = float(rwr) if rwr is not None else 0.5
    closeness = 1.0 - 2.0 * abs(rwr_val - 0.5)
    w_rank = 0.5 + 0.3 * closeness
    w_policy = 1.0 - w_rank
    return w_policy * r_policy + w_rank * r_rank


# ================= v5 reasoning layer =================

def extract_reasoning(text: str) -> str:
    """The text between REASONING: and MOVE: (empty string if absent)."""
    m = re.search(r"REASONING:\s*(.*?)\s*MOVE:", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _word_bigrams(text: str) -> set:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return set(zip(words, words[1:]))


def consistency_term(reasoning: str, move: str) -> float:
    """+CONSISTENCY_BONUS when the chosen coordinate is discussed by name."""
    if not reasoning or not move or move == "PASS":
        return 0.0
    if re.search(rf"\b{re.escape(move)}\b", reasoning, re.IGNORECASE):
        return CONSISTENCY_BONUS
    return 0.0


def length_term(reasoning: str) -> float:
    """0 at/above LENGTH_FULL_CREDIT_CHARS, linear down to LENGTH_MAX_PENALTY."""
    n = len(reasoning)
    if n >= LENGTH_FULL_CREDIT_CHARS:
        return 0.0
    return LENGTH_MAX_PENALTY * (1.0 - n / LENGTH_FULL_CREDIT_CHARS)


def dedup_terms(reasonings: List[str]) -> List[float]:
    """Per-completion penalty for high mean pairwise bigram-Jaccard overlap
    with the *other* completions of the same prompt group."""
    sets = [_word_bigrams(r) for r in reasonings]
    out = []
    for i, si in enumerate(sets):
        others = [s for j, s in enumerate(sets) if j != i]
        if not others or not si:
            out.append(0.0)
            continue
        jacs = [len(si & so) / len(si | so) if (si | so) else 0.0 for so in others]
        mean_jac = sum(jacs) / len(jacs)
        excess = max(0.0, mean_jac - DEDUP_JACCARD_THRESHOLD)
        out.append(DEDUP_MAX_PENALTY * min(1.0, excess / (1.0 - DEDUP_JACCARD_THRESHOLD)))
    return out


def reasoning_quality_rewards(prompts: List[str], completions: List[str]) -> List[float]:
    """Batch version. Completions whose gate fails get exactly 0.0 (the gate
    already penalizes them; stacking here would double-count).

    Grouping: TRL passes each prompt's generations contiguously, so groups are
    recovered by runs of identical prompts.
    """
    n = len(completions)
    moves = [extract_move(c) for c in completions]
    gate_ok = [
        bool(m) and check_spatial_legality(p, m) >= 0
        for p, m in zip(prompts, moves)
    ]
    reasonings = [extract_reasoning(c) for c in completions]

    rewards = [0.0] * n
    for _, grp in groupby(range(n), key=lambda i: prompts[i]):
        group_idx = list(grp)
        group_dedup = dedup_terms([reasonings[i] for i in group_idx])
        for i, dd in zip(group_idx, group_dedup):
            if not gate_ok[i]:
                rewards[i] = 0.0
            else:
                rewards[i] = (
                    consistency_term(reasonings[i], moves[i])
                    + length_term(reasonings[i])
                    + dd
                )
    return rewards
