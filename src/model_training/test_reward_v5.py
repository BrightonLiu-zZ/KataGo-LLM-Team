"""
Unit tests for the v5 reward components (reward_v5.py). Pure CPU, no GPU or
training environment needed:

    python src/model_training/test_reward_v5.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_v5 import (  # noqa: E402
    CONSISTENCY_BONUS,
    LENGTH_FULL_CREDIT_CHARS,
    LENGTH_MAX_PENALTY,
    DEDUP_MAX_PENALTY,
    extract_move,
    extract_reasoning,
    check_spatial_legality,
    consistency_term,
    length_term,
    dedup_terms,
    composite_core,
    reasoning_quality_rewards,
)

BOARD_PROMPT = (
    "[Current 9x9 Board State]\n"
    "   A B C D E F G H J\n"
    " 9 . . . . . . . . .\n"
    " 8 . . . . . . . . .\n"
    " 7 . . . . . . . . .\n"
    " 6 . . . . . . . . .\n"
    " 5 . . . . X . . . .\n"
    " 4 . . . O . . . . .\n"
    " 3 . . . . . . . . .\n"
    " 2 . . . . . . . . .\n"
    " 1 . . . . . . . . .\n"
    "\nPlayer to move: Black.\n"
)

LONG = ("Black holds the center with E5 while White approached from D4. "
        "Extending toward C5 presses the white stone and builds left-side "
        "influence, keeping the initiative for the coming fight over the "
        "lower half of the board.")
assert len(LONG) >= LENGTH_FULL_CREDIT_CHARS, "test fixture must be long enough"


def completion(reasoning, move):
    return f"REASONING: {reasoning}\nMOVE: {move}"


PASSED = 0
FAILED = []


def check(name, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
    else:
        FAILED.append(f"{name} {detail}")
        print(f"  FAIL: {name} {detail}")


# --- extraction ---
check("extract_move basic", extract_move("REASONING: x\nMOVE: C5") == "C5")
check("extract_move pass", extract_move("MOVE: PASS") == "PASS")
check("extract_move missing", extract_move("no move here") is None)
check("extract_reasoning", extract_reasoning(completion("hello world", "C5")) == "hello world")
check("extract_reasoning missing", extract_reasoning("MOVE: C5") == "")

# --- legality (inherited v4 behaviour) ---
check("legal empty", check_spatial_legality(BOARD_PROMPT, "C5") == 0.0)
check("occupied E5", check_spatial_legality(BOARD_PROMPT, "E5") < 0)
check("occupied D4", check_spatial_legality(BOARD_PROMPT, "D4") < 0)
check("pass legal", check_spatial_legality(BOARD_PROMPT, "PASS") == 0.0)

# --- consistency term ---
check("consistency hit", consistency_term("Playing at C5 presses White.", "C5") == CONSISTENCY_BONUS)
check("consistency case-insensitive", consistency_term("playing at c5 works", "C5") == CONSISTENCY_BONUS)
check("consistency miss", consistency_term("A move near the center is good.", "C5") == 0.0)
check("consistency no substring trap", consistency_term("scored 3C5F in hex", "C5") == 0.0)
check("consistency pass neutral", consistency_term("I will pass.", "PASS") == 0.0)
check("consistency empty reasoning", consistency_term("", "C5") == 0.0)

# --- length term ---
check("length full credit", length_term(LONG) == 0.0)
check("length empty worst", abs(length_term("") - LENGTH_MAX_PENALTY) < 1e-9)
half = "x" * (LENGTH_FULL_CREDIT_CHARS // 2)
check("length linear midpoint", abs(length_term(half) - LENGTH_MAX_PENALTY / 2) < 1e-9)
check("length monotone", length_term("x" * 50) < length_term("x" * 150) <= 0.0)

# --- dedup term ---
identical = [LONG] * 4
dd = dedup_terms(identical)
check("dedup identical penalized", all(abs(d - DEDUP_MAX_PENALTY) < 1e-9 for d in dd),
      f"got {dd}")
distinct = [
    "Black should press at C5 to build the left side and attack the lone white stone.",
    "The corner at G3 is the biggest open area; taking it balances the board.",
    "White's D4 stone is weak; capping at D6 restricts its growth toward the top.",
    "A calm extension to E7 secures the upper half and keeps sente for later.",
]
dd = dedup_terms(distinct)
check("dedup distinct unpenalized", all(d == 0.0 for d in dd), f"got {dd}")
check("dedup single completion", dedup_terms(["only one"]) == [0.0])
check("dedup empty reasonings", dedup_terms(["", ""]) == [0.0, 0.0])

# --- composite core sanity (unchanged v4 math) ---
evals = {
    "C5": {"policy": 0.5, "winrate": 0.6, "scoreLead": 2.0},
    "G3": {"policy": 0.01, "winrate": 0.5, "scoreLead": 0.0},
    "D6": {"policy": 0.001, "winrate": 0.4, "scoreLead": -3.0},
}
best = composite_core("C5", evals, 0.5)
worst = composite_core("D6", evals, 0.5)
check("core best beats worst", best > worst, f"{best} vs {worst}")
check("core in [0,1]", 0.0 <= worst <= best <= 1.0)
check("core none move", composite_core(None, evals, 0.5) == 0.0)

# --- batch function: grouping, gating, range ---
prompts = [BOARD_PROMPT] * 4
comps = [
    completion(LONG + " C5 is the point.", "C5"),          # legal, long, mentions move
    completion("short", "C5"),                              # legal, too short
    completion(LONG, "E5"),                                 # occupied -> gate fails -> 0
    "no format at all",                                     # no MOVE -> gate fails -> 0
]
rw = reasoning_quality_rewards(prompts, comps)
check("batch len", len(rw) == 4)
check("gate-failed zero", rw[2] == 0.0 and rw[3] == 0.0, f"got {rw}")
check("good completion positive-ish", rw[0] >= 0.0, f"got {rw[0]}")
check("short penalized", rw[1] < rw[0], f"got {rw}")
check("range bound", all(-0.2 - 1e-9 <= r <= CONSISTENCY_BONUS + 1e-9 for r in rw),
      f"got {rw}")

# two different prompts back-to-back must not cross-contaminate dedup groups
prompts2 = ["P1"] * 2 + ["P2"] * 2
comps2 = [completion(LONG, "C5")] * 2 + [completion(LONG, "C5")] * 2
rw2 = reasoning_quality_rewards(prompts2, comps2)
# within each pair the two reasonings are identical -> dedup fires per group
check("grouping by prompt", all(r < 0.0 + CONSISTENCY_BONUS + 1e-9 for r in rw2))
check("groups independent", rw2[0] == rw2[2] and rw2[1] == rw2[3], f"got {rw2}")

print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    sys.exit(1)
