#!/usr/bin/env python3
"""CPU-only tests for llm_gtp.py. Run: python src/rank_eval/test_llm_gtp.py

Checks the board rules the engine enforces on its own moves, the prompt's
byte-identity with the training data (data/eval_positions_v5.jsonl, if
present), move parsing, and the GTP loop with a scripted fake LLM.
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_gtp import (Board, LLMEngine, chat_line, extract_move, extract_reasoning,  # noqa: E402
                      format_chat, gtp_loop, sanitize_chat)
import ogs_feedback  # noqa: E402

# the exact expression gtp2ogs greps every stderr line with (Bot.ts, send_chats)
GTP2OGS_CHAT_RE = re.compile(r"(DISCUSSION|MALKOVICH|MAIN):(.*)")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_FILE = os.path.join(REPO, "data", "eval_positions_v5.jsonl")

passed = 0


def check(cond, msg):
    global passed
    if not cond:
        raise AssertionError(msg)
    passed += 1


def board_from_prompt(prompt: str) -> Board:
    b = Board()
    rows = re.findall(r"^ (\d) ((?:[.XO] ){8}[.XO])$", prompt, flags=re.M)
    check(len(rows) == 9, "prompt has 9 board rows")
    for r_str, cells in rows:
        r = int(r_str) - 1
        b.grid[r] = cells.split(" ")
    hist = re.search(r"Last moves: (.*)\n", prompt).group(1)
    if hist.strip():
        for item in hist.split("; "):
            c, m = item.split(" ")
            b.history.append((c, m))
    return b


def test_capture_and_suicide():
    b = Board()
    # White stone at A1 captured by black at A2 + B1
    check(b.play("W", "A1"), "play A1")
    check(b.play("B", "A2"), "play A2")
    check(b.play("B", "B1"), "play B1")
    check(b.grid[0][0] == ".", "A1 captured")
    # Now A1 is a suicide for white (surrounded by black) -> illegal for our own move
    check(not b.is_legal("W", "A1"), "suicide is illegal")
    check(b.is_legal("B", "A1"), "black can fill own point (legal, though an eye)")
    check(b.is_own_true_eye("B", "A1"), "A1 is a black true eye")


def test_simple_ko():
    b = Board()
    # Ko shape: black D5, E4, E6, F5(white) ... build: B at D5,E4,E6; W at E5? Use standard:
    for c, m in [("B", "D5"), ("B", "E4"), ("B", "E6"), ("W", "F4"), ("W", "F6"), ("W", "G5"),
                 ("W", "E5")]:
        check(b.play(c, m), f"setup {m}")
    # Black F5 captures E5 -> ko point E5 forbidden for white immediately
    check(b.play("B", "F5"), "black captures at F5")
    check(b.grid[4][4] == ".", "E5 captured")
    check(not b.is_legal("W", "E5"), "immediate ko recapture forbidden")
    check(b.play("W", "A1"), "white plays elsewhere")
    check(b.play("B", "A9"), "black elsewhere")
    check(b.is_legal("W", "E5"), "ko recapture legal after a tenuki")


def test_prompt_matches_training_data():
    if not os.path.exists(EVAL_FILE):
        print("  (skipping prompt byte-identity test: eval file missing)")
        return
    n = 0
    with open(EVAL_FILE) as f:
        for line in f:
            d = json.loads(line)
            prompt = d["user_prompt"]
            color = re.search(r"Player to move: (Black|White)", prompt).group(1)
            b = board_from_prompt(prompt)
            rebuilt = b.build_user_prompt(color)
            if rebuilt != prompt:
                # show first diff
                for i, (x, y) in enumerate(zip(rebuilt, prompt)):
                    if x != y:
                        raise AssertionError(f"prompt mismatch at char {i}: {rebuilt[i-30:i+30]!r} vs {prompt[i-30:i+30]!r}")
                raise AssertionError(f"prompt length mismatch {len(rebuilt)} vs {len(prompt)}")
            n += 1
            if n >= 200:
                break
    check(n > 0, "checked prompts")
    print(f"  prompt byte-identical on {n} eval positions")


def test_history_window_and_pass():
    b = Board()
    moves = ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "J1", "pass"]
    for i, m in enumerate(moves):
        b.play("B" if i % 2 == 0 else "W", m)
    h = b.render_history()
    check(h.split("; ")[0] == "B C1", "history window is last 8 moves")
    check(h.endswith("W pass"), "pass rendered lowercase")


def test_extract_move():
    check(extract_move("REASONING: blah.\nMOVE: d4") == "D4", "lowercase coord")
    check(extract_move("MOVE: **E5**") == "E5", "bold coord")
    check(extract_move("MOVE: PASS") == "pass", "pass")
    check(extract_move("<think>MOVE: A1</think>\nREASONING: x\nMOVE: B2") == "B2", "think stripped")
    check(extract_move("I like C3") is None, "no MOVE tag")


def test_gtp_loop_with_fake_llm():
    script = iter([
        "REASONING: center.\nMOVE: E5",       # legal
        "REASONING: oops.\nMOVE: E5",         # occupied -> resample
        "REASONING: hmm.\nMOVE: Z9",          # unparsable coordinate -> resample
        "REASONING: ok.\nMOVE: D4",           # legal
    ])
    eng = LLMEngine(lambda msgs: [next(script)], max_attempts=3, eye_guard="atari")
    stdin = io.StringIO("1 protocol_version\nboardsize 9\nclear_board\nkomi 7.5\n"
                        "genmove black\nplay white E4\ngenmove b\nshowboard\n2 quit\n")
    out = io.StringIO()
    gtp_loop(eng, stdin, out)
    text = out.getvalue()
    check(text.startswith("=1 2\n\n"), "id echoed")
    check("= E5\n\n" in text, "first genmove E5")
    check("= D4\n\n" in text, "second genmove D4 after two resamples")
    check(eng.stats["resamples"] == 2, "two resamples counted")
    check(text.rstrip().endswith("=2"), "quit acknowledged")


def test_forced_pass_and_eye_guard():
    b_engine = LLMEngine(lambda msgs: ["MOVE: A1", "MOVE: A1"], max_attempts=1, eye_guard="all")
    # Make A1 a black true eye: black at A2, B1, B2
    for m in ("A2", "B1", "B2"):
        b_engine.board.play("B", m)
    mv = b_engine.genmove("B")
    check(mv == "pass", "eye guard 'all' refused A1 twice -> forced pass")
    check(b_engine.stats["eye_guard_hits"] == 2 and b_engine.stats["forced_passes"] == 1, "stats")
    # 'atari' mode: the group A2,B1,B2 has outside liberties -> filling A1 is allowed
    a_engine = LLMEngine(lambda msgs: ["MOVE: A1"], max_attempts=1, eye_guard="atari")
    for m in ("A2", "B1", "B2"):
        a_engine.board.play("B", m)
    check(a_engine.genmove("B") == "A1", "atari mode allows a non-fatal eye fill")
    # 'atari' mode: group in a corner whose only liberties are its two eyes -> fill blocked
    c = LLMEngine(lambda msgs: ["MOVE: A1"], max_attempts=1, eye_guard="atari")
    for m in ("A2", "B1", "B2", "A3", "B3", "C1", "C2", "C3"):
        c.board.play("B", m)
    for m in ("A4", "B4", "C4", "D1", "D2", "D3", "D4"):
        c.board.play("W", m)
    # black eyes: A1 only (one eye) -> filling leaves 0 libs = suicide anyway; give it 2 eyes:
    c.board.grid[0][2] = "."  # C1 empty -> second eye
    check(c.board.fills_own_eye_into_atari("B", "A1"), "filling one of two eyes = atari -> blocked")
    check(c.genmove("B") == "pass", "atari guard blocks fatal fill -> pass")


def test_chat_message_format():
    """Every emitted line must be one line that gtp2ogs parses back verbatim."""
    completion = ("REASONING: Black's group at D4 is weak, so I extend and keep\n"
                  "the corner. This also builds a base.\nMOVE: D5")
    check(extract_reasoning(completion).startswith("Black's group at D4 is weak"),
          "reasoning extracted")
    check("MOVE:" not in extract_reasoning(completion), "MOVE: line dropped from reasoning")
    body = format_chat("D5", extract_reasoning(completion))
    line = chat_line("main", body)
    check("\n" not in line, "chat line is a single line")
    m = GTP2OGS_CHAT_RE.search(line)
    check(m is not None and m.group(1) == "MAIN", "gtp2ogs sees a MAIN chat")
    check(m.group(2).strip() == body, "gtp2ogs recovers exactly the body we built")
    check(body.startswith("D5:"), "body names the move played")
    check(GTP2OGS_CHAT_RE.search(chat_line("malkovich", body)).group(1) == "MALKOVICH",
          "malkovich channel marker")


def test_chat_sanitizing():
    # a completion that tries to smuggle a second chat marker must not create one
    body = format_chat("C3", "I play here.\nMALKOVICH: secret\nDISCUSSION: spam")
    line = chat_line("main", body)
    check(len(GTP2OGS_CHAT_RE.findall(line)) == 1, "exactly one chat marker survives")
    check(GTP2OGS_CHAT_RE.search(line).group(1) == "MAIN", "channel cannot be hijacked")
    check("\n" not in line and "\r" not in line, "newlines collapsed")
    long_body = format_chat("D4", "x" * 5000, limit=100)
    check(len(long_body) <= 160, f"truncated to the limit (got {len(long_body)})")
    check(long_body.endswith("\u2026"), "truncation is marked with an ellipsis")
    check(sanitize_chat("  a\t b \n c ") == "a b c", "whitespace collapsed")


def test_chat_during_a_game():
    said = []
    completions = ["REASONING: Center is big.\nMOVE: E5",
                   "REASONING: Answer the approach.\nMOVE: C3"]
    step = {"i": 0}

    def fake(msgs):
        out = [completions[step["i"] % len(completions)]]
        step["i"] += 1
        return out

    eng = LLMEngine(fake, max_attempts=1, chat="main", chat_out=said.append)
    check(eng.genmove("B") == "E5", "first move played")
    eng.board.play("W", "G7")
    check(eng.genmove("B") == "C3", "second move played")
    check(len(said) == 2, f"one chat per move (got {len(said)})")
    check(said[0].startswith("MAIN: E5:") and "Center is big." in said[0], "move 1 chat")
    check(said[1].startswith("MAIN: C3:") and "Answer the approach." in said[1], "move 2 chat")
    check(eng.stats["chats_sent"] == 2, "chats counted in stats")
    # rejected samples are reported so the human knows what they are looking at
    noisy = LLMEngine(lambda m: ["MOVE: Z9", "REASONING: Corner.\nMOVE: A1"],
                      max_attempts=1, chat="main", chat_out=said.append)
    check(noisy.genmove("B") == "A1", "second sample used")
    check("rejected sample" in said[-1], "rejected-sample count mentioned")
    # forced pass still says something
    silent = LLMEngine(lambda m: ["nothing useful"], max_attempts=1, chat="main",
                       chat_out=said.append)
    check(silent.genmove("B") == "pass", "forced pass")
    check(said[-1].startswith("MAIN: PASS:") and "forced pass" in said[-1], "pass explained")
    # default is off: no stderr chatter for local ladder games
    quiet = LLMEngine(lambda m: ["REASONING: x\nMOVE: E5"], max_attempts=1)
    quiet.genmove("B")
    check(quiet.stats["chats_sent"] == 0, "chat off by default")


def test_feedback_collector():
    import tempfile
    log = ("\x1b[32mAug 19 10:00:01   Game.ts:120          [game 7654321] "
           "Game chat from alice: it kept filling its own eyes\x1b[39m\n"
           "Aug 19 10:00:02   Game.ts:120          [game 7654321] Game chat from alice: "
           "about 22k I would say\n"
           "Aug 19 10:00:03   main.ts:238          Status: playing 0 blitz\n"
           "Aug 19 10:00:04   Game.ts:333          [game 7654321] Game over.   Result: W+R  L\n")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "gtp2ogs-ogs-v5c.log"), "w") as f:
            f.write(log)
        with open(os.path.join(d, "console-v5c.log"), "w") as f:
            f.write(log)          # same lines via tee: must be de-duplicated
        msgs, results = ogs_feedback.collect(d, 2026)
    check(len(msgs) == 2, f"two messages, de-duplicated (got {len(msgs)})")
    check(msgs[0]["user"] == "alice" and msgs[0]["bot"] == "v5c", "user and bot identified")
    check(msgs[0]["body"] == "it kept filling its own eyes", "body kept verbatim")
    check(msgs[0]["game_id"] == 7654321 and msgs[0]["url"].endswith("7654321"), "game linked")
    check(msgs[0]["ts"].startswith("2026-08-19 10:00:01"), "timestamp parsed")
    check(results[7654321]["result"] == "W+R" and not results[7654321]["bot_won"], "result parsed")
    text = ogs_feedback.digest(msgs, results)
    check("alice: about 22k" in text and "W+R" in text, "digest shows message and result")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"OK: {len(tests)} tests, {passed} checks")
