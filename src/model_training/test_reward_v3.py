"""
Unit tests for train_grpo_v3 reward functions.
Run from repo root: python src/model_training/test_reward_v3.py
No GPU required.
"""
import sys
import os
import unittest.mock as mock

for mod in ['datasets', 'transformers', 'peft', 'trl', 'torch']:
    sys.modules[mod] = mock.MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import train_grpo_v3 as rw

EVALS = {
    "D6": {"policy": 0.38,  "winrate": 0.55, "scoreLead":  2.0},
    "F4": {"policy": 0.25,  "winrate": 0.50, "scoreLead": -1.0},
    "E5": {"policy": 0.10,  "winrate": 0.45, "scoreLead": -3.0},
    "A1": {"policy": 0.001, "winrate": 0.30, "scoreLead":-10.0},
    "B9": {"policy": 0.0002,"winrate": None, "scoreLead": None},
}
EVALS_ALL_MISSING = {"F4": {"policy": 0.25, "winrate": None, "scoreLead": None}}

PROMPT = (
    "   A B C D E F G H J\n"
    " 9 . . . . . . . . .\n"
    " 8 . . . . . . . . .\n"
    " 7 . . . . . . . . .\n"
    " 6 . . . X O . . . .\n"
    " 5 . . . O X . . . .\n"
    " 4 . . . . . . . . .\n"
    " 3 . . . . . . . . .\n"
    " 2 . . . . . . . . .\n"
    " 1 . . . . . . . . .\n"
    "\nPlayer to move: Black."
)
GOOD = (
    "<think>\nStep 1: F4 open.\nStep 2: F4 vs E3. F4 wins.\nStep 3: F4 is empty.\n</think>\n"
    "MOVE: F4\nEXPLAIN: Good move."
)
OCC  = "<think>\nD6 looks interesting.\n</think>\nMOVE: D6\nEXPLAIN: Playing D6."
BAD  = "I think somewhere is good."
INCON = "<think>\nF4 is the best move here.\n</think>\nMOVE: D6\nEXPLAIN: Actually D6."


def check(name, got, exp, tol=0.01):
    ok = abs(got - exp) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {got:.4f} (expected {exp:.4f})")
    return ok


all_ok = True

print("=" * 55)
print("_compute_r_policy")
print("=" * 55)
all_ok &= check("D6 (best)",    rw._compute_r_policy("D6", EVALS), 0.38)
all_ok &= check("F4",           rw._compute_r_policy("F4", EVALS), 0.25)
all_ok &= check("A1 (bad)",     rw._compute_r_policy("A1", EVALS), 0.001)
all_ok &= check("H9 (missing)", rw._compute_r_policy("H9", EVALS), 0.0)

print("=" * 55)
print("_compute_r_score  (best_scoreLead=2.0)")
print("=" * 55)
all_ok &= check("D6 → 1.0",   rw._compute_r_score("D6", EVALS),  1.0)
all_ok &= check("F4 → 0.25",  rw._compute_r_score("F4", EVALS),  0.25)
all_ok &= check("E5 → -0.25", rw._compute_r_score("E5", EVALS), -0.25)
all_ok &= check("A1 → -1.0",  rw._compute_r_score("A1", EVALS), -1.0)
all_ok &= check("B9 → -1.0",  rw._compute_r_score("B9", EVALS), -1.0)
all_ok &= check("H9 → -1.0",  rw._compute_r_score("H9", EVALS), -1.0)
all_ok &= check("all missing", rw._compute_r_score("F4", EVALS_ALL_MISSING), -1.0)

print("=" * 55)
print("_compute_r_winrate  (best_winrate=0.55)")
print("=" * 55)
all_ok &= check("D6 → 1.0",  rw._compute_r_winrate("D6", EVALS), 1.0)
all_ok &= check("F4 → 0.9",  rw._compute_r_winrate("F4", EVALS), 0.9)
all_ok &= check("E5 → 0.8",  rw._compute_r_winrate("E5", EVALS), 0.8)
all_ok &= check("A1 → 0.5",  rw._compute_r_winrate("A1", EVALS), 0.5)
all_ok &= check("B9 → -1.0", rw._compute_r_winrate("B9", EVALS), -1.0)
all_ok &= check("H9 → -1.0", rw._compute_r_winrate("H9", EVALS), -1.0)

print("=" * 55)
print("format_and_legality_reward_func")
print("=" * 55)
r = rw.format_and_legality_reward_func([PROMPT], [GOOD])[0]
all_ok &= check("F4 (empty) → +0.2",   r, 0.2)
r = rw.format_and_legality_reward_func([PROMPT], [OCC])[0]
all_ok &= check("D6 (occupied) → -0.6",r, -0.6)
r = rw.format_and_legality_reward_func([PROMPT], [BAD])[0]
all_ok &= check("no move → -1.0",       r, -1.0)

print("=" * 55)
print("katago_composite_reward_func")
print("=" * 55)
# F4: r_p=0.25, r_s=0.25, r_w=0.9 → 0.5*0.25+0.3*0.25+0.2*0.9 = 0.38
r = rw.katago_composite_reward_func([PROMPT], [GOOD], [EVALS])[0]
all_ok &= check("F4 full data → 0.38",    r, 0.38)
r = rw.katago_composite_reward_func([PROMPT], [OCC],  [EVALS])[0]
all_ok &= check("occupied → 0.0",          r, 0.0)
r = rw.katago_composite_reward_func([PROMPT], [BAD],  [EVALS])[0]
all_ok &= check("no move → 0.0",           r, 0.0)
# F4 missing: 0.5*0.25 + 0.3*(-1) + 0.2*(-1) = -0.375
r = rw.katago_composite_reward_func([PROMPT], [GOOD], [EVALS_ALL_MISSING])[0]
all_ok &= check("F4 missing data → -0.375",r, -0.375)

print("=" * 55)
print("thinking_quality_reward_func")
print("=" * 55)
r = rw.thinking_quality_reward_func([PROMPT], [GOOD],  [EVALS])[0]
all_ok &= check("consistent+in_evals → +0.5", r, 0.5)
r = rw.thinking_quality_reward_func([PROMPT], [OCC],   [EVALS])[0]
all_ok &= check("occupied → 0.0",              r, 0.0)
r = rw.thinking_quality_reward_func([PROMPT], [INCON], [EVALS])[0]
all_ok &= check("inconsistent+occupied → 0.0", r, 0.0)

print()
print("=" * 55)
print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
print("=" * 55)
sys.exit(0 if all_ok else 1)
