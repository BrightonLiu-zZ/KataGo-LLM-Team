#!/usr/bin/env python3
"""GTP engine that plays 9x9 Go by asking the fine-tuned LLM for moves.

Linux counterpart of ``src/interception/gtp_proxy_v4.cpp`` for headless use
(OGS via gtp2ogs, gogui-twogtp ladders, CGOS). Talks to any OpenAI-compatible
chat endpoint (vLLM ``vllm serve``, llama-server, LM Studio).

Design notes (see docs/RANK-EVAL-PLAN.md §4):

* The prompt is byte-identical to the v4/v5 training prompt: ASCII board,
  **last 8 moves** (``game_moves[-8:]`` in prepare_training_data.py, passes
  rendered as ``B pass``), "Player to move", "Valid empty coordinates" sorted
  row-major from row 1 — the same order ``prepare_v4_dataset.py`` used.
* The engine keeps its own board (captures, suicide, simple ko) because GTP
  ``play`` reports only the stone placed, never the captures (ADR 0004).
* Each request samples ``--n-samples`` completions; the first legal one (in
  sample order) is played, which equals "resample until legal" at the cost of
  one batched request. After ``--max-attempts`` requests without a legal move
  the engine passes. ``--valid-list legal`` (default) drops suicide/ko points
  from the "Valid empty coordinates" list — the training gate only checked
  emptiness, so the model happily proposes suicides otherwise (measured: 30%
  of moves ended in a forced pass vs GnuGo with the exact training list);
  ``--valid-list empty`` reproduces the training prompt byte-for-byte.
  ``--eye-guard atari`` (default) refuses only eye-fills that leave the group
  in atari (the model does this; blocking *every* eye-fill was measured to be
  worse — it insisted on the point and passed instead). Everything is logged
  per move so a report can state exactly how often each guard fired.
* Passing/resigning is left to gtp2ogs' ``ending_bot`` (GnuGo) — this engine
  never resigns and passes only when it cannot produce a legal move.

Usage: ``python llm_gtp.py --model go-v5c --base-url http://127.0.0.1:8100/v1``
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

BOARD_SIZE = 9
VALID_COLS = "ABCDEFGHJ"  # keep consistent with training scripts (no I)
EMPTY, BLACK, WHITE = ".", "X", "O"

SYSTEM_PROMPT = (
    "You are a 9x9 Go (Weiqi) player. "
    "Board notation: '.' = empty, 'X' = Black, 'O' = White. "
    "Columns: A-J (no I). Rows: 1 (bottom) to 9 (top). "
    "You MUST only play on an empty '.' intersection listed in the valid coordinates.\n\n"
    "Respond in exactly this format:\n"
    "REASONING: [2-4 sentences analyzing the position]\n"
    "MOVE: [coordinate, e.g. D4]"
)

MOVE_RE = re.compile(r"MOVE:\s*\**\s*([A-HJa-hj][1-9]|PASS|pass)", re.IGNORECASE)
HISTORY_WINDOW = 8


# ----------------------------------------------------------------------------
# Board
# ----------------------------------------------------------------------------
class Board:
    """9x9 board with capture / suicide / simple-ko handling."""

    def __init__(self) -> None:
        self.grid: List[List[str]] = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.history: List[Tuple[str, str]] = []  # ("B"/"W", "D4"/"pass")
        self._snapshots: List[Tuple[List[List[str]], List[Tuple[str, str]], Optional[Tuple]]] = []
        self.ko_point: Optional[Tuple[int, int]] = None  # (row, col) forbidden for the next move

    # ---- coordinates ----
    @staticmethod
    def parse_coord(coord: str) -> Optional[Tuple[int, int]]:
        coord = coord.strip().upper()
        if len(coord) != 2 or coord[0] not in VALID_COLS or not coord[1].isdigit():
            return None
        col = VALID_COLS.index(coord[0])
        row = int(coord[1]) - 1
        if not (0 <= row < BOARD_SIZE):
            return None
        return row, col

    @staticmethod
    def coord_str(row: int, col: int) -> str:
        return f"{VALID_COLS[col]}{row + 1}"

    # ---- group logic ----
    def _neighbors(self, r: int, c: int):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                yield nr, nc

    def _group(self, r: int, c: int, grid=None) -> Tuple[List[Tuple[int, int]], int]:
        grid = grid or self.grid
        color = grid[r][c]
        stack, seen, libs = [(r, c)], {(r, c)}, set()
        while stack:
            cr, cc = stack.pop()
            for nr, nc in self._neighbors(cr, cc):
                v = grid[nr][nc]
                if v == EMPTY:
                    libs.add((nr, nc))
                elif v == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return list(seen), len(libs)

    def _simulate(self, r: int, c: int, stone: str):
        """Return (new_grid, captured_count) or None if the move is illegal."""
        if self.grid[r][c] != EMPTY:
            return None
        grid = copy.deepcopy(self.grid)
        grid[r][c] = stone
        opp = WHITE if stone == BLACK else BLACK
        captured = 0
        for nr, nc in self._neighbors(r, c):
            if grid[nr][nc] == opp:
                group, libs = self._group(nr, nc, grid)
                if libs == 0:
                    for gr, gc in group:
                        grid[gr][gc] = EMPTY
                    captured += len(group)
        _, own_libs = self._group(r, c, grid)
        if own_libs == 0:
            return None  # suicide
        return grid, captured

    def is_legal(self, color: str, coord: str) -> bool:
        pt = self.parse_coord(coord)
        if pt is None:
            return False
        if self.ko_point == pt:
            return False
        stone = BLACK if color.upper().startswith("B") else WHITE
        return self._simulate(pt[0], pt[1], stone) is not None

    def play(self, color: str, coord: str, strict: bool = False) -> bool:
        """Apply a move. ``coord`` may be 'pass'. Returns False if illegal.

        With strict=False (moves coming from the server/opponent) an occupied
        point or ko violation is still applied as far as possible so that the
        board tracks the server; suicide simply removes the group like the C++
        proxy did. With strict=True (our own candidate) any illegality fails.
        """
        short = "B" if color.upper().startswith("B") else "W"
        stone = BLACK if short == "B" else WHITE
        self._snapshots.append((copy.deepcopy(self.grid), list(self.history), self.ko_point))
        if coord.strip().lower() == "pass":
            self.history.append((short, "pass"))
            self.ko_point = None
            return True
        pt = self.parse_coord(coord)
        if pt is None:
            self._snapshots.pop()
            return False
        r, c = pt
        if strict and (self.ko_point == pt):
            self._snapshots.pop()
            return False
        sim = self._simulate(r, c, stone)
        if sim is None:
            if strict or self.grid[r][c] != EMPTY:
                self._snapshots.pop()
                return False
            # lenient suicide (should never come from OGS/Japanese rules)
            self.grid[r][c] = stone
            group, _ = self._group(r, c)
            for gr, gc in group:
                self.grid[gr][gc] = EMPTY
            self.history.append((short, self.coord_str(r, c)))
            self.ko_point = None
            return True
        new_grid, captured = sim
        # simple ko: single stone captured, and the new stone is alone with one liberty
        new_ko = None
        if captured == 1:
            group, libs = self._group(r, c, new_grid)
            if len(group) == 1 and libs == 1:
                for nr, nc in self._neighbors(r, c):
                    if new_grid[nr][nc] == EMPTY and self.grid[nr][nc] != EMPTY:
                        new_ko = (nr, nc)
        self.grid = new_grid
        self.ko_point = new_ko
        self.history.append((short, self.coord_str(r, c)))
        return True

    def undo(self) -> bool:
        if not self._snapshots:
            return False
        self.grid, self.history, self.ko_point = self._snapshots.pop()
        return True

    def clear(self) -> None:
        self.__init__()

    # ---- eye guard ----
    def fills_own_eye_into_atari(self, color: str, coord: str) -> bool:
        """True if ``coord`` is an own true eye AND filling it leaves the
        resulting group with <=1 liberty (i.e. the fill kills the group)."""
        if not self.is_own_true_eye(color, coord):
            return False
        pt = self.parse_coord(coord)
        stone = BLACK if color.upper().startswith("B") else WHITE
        sim = self._simulate(pt[0], pt[1], stone)
        if sim is None:
            return True
        _, libs = self._group(pt[0], pt[1], sim[0])
        return libs <= 1

    def is_own_true_eye(self, color: str, coord: str) -> bool:
        pt = self.parse_coord(coord)
        if pt is None or self.grid[pt[0]][pt[1]] != EMPTY:
            return False
        r, c = pt
        stone = BLACK if color.upper().startswith("B") else WHITE
        opp = WHITE if stone == BLACK else BLACK
        for nr, nc in self._neighbors(r, c):
            if self.grid[nr][nc] != stone:
                return False
        diag_opp = 0
        diag_total = 0
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                diag_total += 1
                if self.grid[nr][nc] == opp:
                    diag_opp += 1
        # interior: at most one opponent diagonal; edge/corner: none
        return diag_opp == 0 if diag_total < 4 else diag_opp <= 1

    # ---- rendering (byte-identical to training prompt) ----
    def render_ascii(self) -> str:
        lines = ["[Current 9x9 Board State]", "   A B C D E F G H J"]
        for r in range(BOARD_SIZE - 1, -1, -1):
            lines.append(f" {r + 1} " + " ".join(self.grid[r]))
        return "\n".join(lines)

    def render_history(self) -> str:
        recent = self.history[-HISTORY_WINDOW:]
        return "; ".join(f"{c} {m}" for c, m in recent)

    def valid_coords(self, color: Optional[str] = None) -> List[str]:
        """Empty points in training order (row 1 first). With ``color`` given,
        only points that are *legal* for that colour (no suicide, no ko) —
        see LLMEngine(valid_list="legal")."""
        pts = [
            self.coord_str(r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if self.grid[r][c] == EMPTY
        ]
        if color is None:
            return pts
        return [p for p in pts if self.is_legal(color, p)]

    def build_user_prompt(self, color: str, valid_list: str = "empty") -> str:
        full = "Black" if color.upper().startswith("B") else "White"
        coords = self.valid_coords(color if valid_list == "legal" else None)
        return (
            f"{self.render_ascii()}\n\n"
            f"Last moves: {self.render_history()}\n\n"
            f"Player to move: {full}. \n"
            f"Analyze the board, then pick a move and explain your reasoning.\n\n"
            f"Valid empty coordinates: [{', '.join(coords)}]"
        )

    def to_sgf(self, komi: float, black_name: str, white_name: str, result: str = "") -> str:
        def sgf_pt(coord: str) -> str:
            if coord == "pass":
                return ""
            pt = self.parse_coord(coord)
            assert pt is not None
            r, c = pt
            return "abcdefghi"[c] + "abcdefghi"[BOARD_SIZE - 1 - r]

        moves = "".join(f";{c}[{sgf_pt(m)}]" for c, m in self.history)
        date = _dt.date.today().isoformat()
        re_prop = f"RE[{result}]" if result else ""
        return (
            f"(;GM[1]FF[4]SZ[9]KM[{komi}]DT[{date}]PB[{black_name}]PW[{white_name}]{re_prop}"
            f"{moves})\n"
        )


# ----------------------------------------------------------------------------
# LLM client
# ----------------------------------------------------------------------------
class OpenAIChatClient:
    def __init__(self, base_url: str, model: str, temperature: float, top_p: float,
                 max_tokens: int, timeout: float = 120.0, api_key: str = "EMPTY", n: int = 1) -> None:
        import requests  # local import so tests can run without it

        self._requests = requests
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.n = n
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def complete(self, messages: List[Dict[str, str]]) -> List[str]:
        """Return ``n`` sampled completions (one request, batched server-side)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "chat_template_kwargs": {"enable_thinking": False},  # ADR 0003
        }
        resp = self._requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for choice in data["choices"]:
            msg = choice["message"]
            content = msg.get("content") or ""
            if not content and msg.get("reasoning_content"):
                content = msg["reasoning_content"]  # LM Studio-style routing
            out.append(content)
        return out


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_move(text: str) -> Optional[str]:
    m = MOVE_RE.search(strip_think(text))
    if not m:
        return None
    mv = m.group(1).upper()
    return "pass" if mv == "PASS" else mv


# ----------------------------------------------------------------------------
# Game chat (gtp2ogs picks these up from stderr)
# ----------------------------------------------------------------------------
# gtp2ogs scans every stderr line with /(DISCUSSION|MALKOVICH|MAIN):(.*)/ when
# ``bot.send_chats`` is on, and posts the captured group to the game chat
# ("main" = both players see it, "malkovich" = hidden from the opponent until
# the game ends).  Everything after the marker up to the newline is the
# message, so a chat line must be exactly one line, and any *further* marker
# inside the text would be re-interpreted -- hence the scrubbing below.
CHAT_MARKER_RE = re.compile(r"(DISCUSSION|MALKOVICH|MAIN)\s*:", re.IGNORECASE)
REASONING_RE = re.compile(r"REASONING:\s*(.*?)(?:\n\s*MOVE:|$)", re.IGNORECASE | re.DOTALL)
CHAT_LIMIT = 700   # the model's REASONING is typically 300-600 chars; truncation should be rare


def extract_reasoning(text: str) -> str:
    """The ``REASONING:`` part of a completion, without the ``MOVE:`` line."""
    body = strip_think(text)
    m = REASONING_RE.search(body)
    if m:
        return m.group(1).strip()
    return MOVE_RE.split(body)[0].strip()


def sanitize_chat(text: str, limit: int = CHAT_LIMIT) -> str:
    """One line, no chat markers, at most ``limit`` characters."""
    one_line = " ".join(str(text).split())
    one_line = CHAT_MARKER_RE.sub(lambda m: m.group(1).lower() + " -", one_line)
    if len(one_line) > limit:
        one_line = one_line[: max(1, limit - 1)].rstrip() + "\u2026"
    return one_line


def format_chat(move: str, reasoning: str, *, rejected: int = 0, note: str = "",
                limit: int = CHAT_LIMIT) -> str:
    """The message body for one move; ``chat_line()`` adds the channel marker."""
    head = f"{move.upper()}:" if move != "pass" else "PASS:"
    parts = [head, sanitize_chat(reasoning, limit) or "(no reasoning returned)"]
    if note:
        parts.append(f"[{note}]")
    if rejected:
        parts.append(f"[{rejected} rejected sample{'s' if rejected > 1 else ''}]")
    return sanitize_chat(" ".join(parts), limit + 60)


def chat_line(channel: str, body: str) -> str:
    """``MAIN: ...`` / ``MALKOVICH: ...`` -- exactly what gtp2ogs greps for."""
    marker = "MALKOVICH" if channel == "malkovich" else "MAIN"
    return f"{marker}: {body}"


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
class LLMEngine:
    NAME = "llm-go-9x9"
    VERSION = "0.1"

    def __init__(self, complete: Callable[[List[Dict[str, str]]], str], *, max_attempts: int = 3,
                 eye_guard: str = "atari", valid_list: str = "legal", log_dir: Optional[str] = None,
                 label: str = "llm", model_name: str = "",
                 log: Callable[[str], None] = lambda s: None,
                 chat: str = "off", chat_limit: int = CHAT_LIMIT,
                 chat_out: Optional[Callable[[str], None]] = None) -> None:
        self.complete = complete
        self.max_attempts = max_attempts
        self.eye_guard = eye_guard
        self.valid_list = valid_list
        self.log_dir = log_dir
        self.label = label
        self.model_name = model_name
        self.log = log
        self.board = Board()
        self.komi = 7.5
        self.game_id = self._new_game_id()
        self.chat = chat
        self.chat_limit = chat_limit
        self.chat_out = chat_out if chat_out is not None else self._default_chat_out
        self.stats = {"genmoves": 0, "resamples": 0, "eye_guard_hits": 0, "forced_passes": 0,
                      "chats_sent": 0}
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # ---- game bookkeeping ----
    @staticmethod
    def _new_game_id() -> str:
        return _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    def _write_sgf(self) -> None:
        if not self.log_dir:
            return
        path = os.path.join(self.log_dir, f"{self.label}-{self.game_id}.sgf")
        with open(path, "w") as f:
            f.write(self.board.to_sgf(self.komi, "?", "?"))

    def _log_move(self, record: Dict) -> None:
        if not self.log_dir:
            return
        record.update({"game_id": self.game_id, "label": self.label, "model": self.model_name,
                       "ts": time.time()})
        with open(os.path.join(self.log_dir, f"{self.label}-moves.jsonl"), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- game chat ----
    @staticmethod
    def _default_chat_out(line: str) -> None:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()

    def say(self, body: str) -> None:
        """Post one line to the game chat (no-op unless --chat is on)."""
        if self.chat == "off" or not body:
            return
        self.chat_out(chat_line(self.chat, sanitize_chat(body, self.chat_limit + 60)))
        self.stats["chats_sent"] += 1

    def _chat_for_move(self, move: str, attempts: List[Dict]) -> None:
        if self.chat == "off":
            return
        played = next((a for a in attempts if a.get("verdict") in ("ok", "pass")), None)
        rejected = sum(1 for a in attempts if a.get("verdict") not in ("ok", "pass", None)
                       and "completion" in a)
        if played is None:
            self.say(format_chat(move, "no legal move in any sample", note="forced pass",
                                 rejected=rejected, limit=self.chat_limit))
            return
        self.say(format_chat(move, extract_reasoning(played.get("completion", "")),
                             rejected=rejected, limit=self.chat_limit))

    # ---- move generation ----
    def genmove(self, color: str) -> str:
        self.stats["genmoves"] += 1
        user_prompt = self.board.build_user_prompt(color, self.valid_list)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}]
        attempts = []
        chosen = None
        for _round in range(self.max_attempts):
            t0 = time.time()
            try:
                completions = self.complete(messages)
            except Exception as e:  # network / server error
                attempts.append({"error": repr(e), "latency": time.time() - t0})
                self.log(f"LLM call failed: {e!r}")
                time.sleep(1.0)
                continue
            latency = time.time() - t0
            if isinstance(completions, str):
                completions = [completions]
            # samples are examined in order; the first legal one is played, so the
            # result equals "resample until legal" but costs one batched request
            for completion in completions:
                move = extract_move(completion)
                verdict = "ok"
                if move is None:
                    verdict = "no_move"
                elif move == "pass":
                    verdict = "pass"
                elif not self.board.is_legal(color, move):
                    verdict = "illegal"
                elif self.eye_guard == "all" and self.board.is_own_true_eye(color, move):
                    verdict = "eye_guard"
                    self.stats["eye_guard_hits"] += 1
                elif self.eye_guard == "atari" and self.board.fills_own_eye_into_atari(color, move):
                    verdict = "eye_guard"
                    self.stats["eye_guard_hits"] += 1
                attempts.append({"completion": completion, "move": move, "verdict": verdict,
                                 "latency": round(latency, 3)})
                if verdict in ("ok", "pass"):
                    chosen = move
                    break
                self.stats["resamples"] += 1
            if chosen is not None:
                break
        if chosen is None:
            chosen = "pass"
            self.stats["forced_passes"] += 1
        self.board.play(color, chosen, strict=False)
        self._chat_for_move(chosen, attempts)
        self._log_move({"color": color, "move": chosen, "attempts": attempts,
                        "move_number": len(self.board.history), "prompt": user_prompt})
        self._write_sgf()
        return chosen

    # ---- GTP dispatch ----
    COMMANDS = ["protocol_version", "name", "version", "known_command", "list_commands", "quit",
                "boardsize", "clear_board", "komi", "play", "genmove", "undo", "showboard",
                "fixed_handicap", "place_free_handicap", "set_free_handicap",
                "time_settings", "time_left", "kgs-time_settings", "llm-stats"]

    # Standard 9x9 handicap points, in the order the GTP spec prescribes.
    HANDICAP_POINTS = ["G7", "C3", "G3", "C7", "E5", "C5", "G5", "E3", "E7"]

    def _place_handicap(self, vertices: List[str]) -> bool:
        for v in vertices:
            if not self.board.play("b", v, strict=False):
                return False
        self._write_sgf()
        return True

    def _fixed_handicap(self, n: int) -> Optional[List[str]]:
        """The n standard points, in the fixed-handicap arrangement (GTP 2 order)."""
        if n < 2 or n > 9:
            return None
        pts = self.HANDICAP_POINTS[:4] if n >= 4 else self.HANDICAP_POINTS[:n]
        if n == 5 or n == 7 or n == 9:
            extra = {5: ["E5"], 7: ["C5", "G5"], 9: ["C5", "G5", "E3", "E7"]}
            pts = self.HANDICAP_POINTS[:4] + extra[n]
        elif n == 6:
            pts = self.HANDICAP_POINTS[:4] + ["C5", "G5"]
        elif n == 8:
            pts = self.HANDICAP_POINTS[:4] + ["C5", "G5", "E3", "E7"]
        return pts[:n]

    def handle(self, line: str) -> Tuple[bool, str, bool]:
        """Returns (ok, response, quit)."""
        parts = line.strip().split()
        if not parts:
            return True, "", False
        cmd, args = parts[0].lower(), parts[1:]
        if cmd == "protocol_version":
            return True, "2", False
        if cmd == "name":
            return True, self.NAME, False
        if cmd == "version":
            return True, f"{self.VERSION} ({self.model_name})", False
        if cmd == "known_command":
            return True, "true" if args and args[0] in self.COMMANDS else "false", False
        if cmd == "list_commands":
            return True, "\n".join(self.COMMANDS), False
        if cmd == "quit":
            return True, "", True
        if cmd == "boardsize":
            if not args or args[0] != str(BOARD_SIZE):
                return False, "unacceptable size", False
            self.board.clear()
            return True, "", False
        if cmd == "clear_board":
            self.board.clear()
            self.game_id = self._new_game_id()
            return True, "", False
        if cmd == "komi":
            try:
                self.komi = float(args[0])
            except (IndexError, ValueError):
                return False, "syntax error", False
            return True, "", False
        if cmd == "play":
            if len(args) < 2:
                return False, "syntax error", False
            color, vertex = args[0], args[1]
            if color.lower() not in ("b", "w", "black", "white"):
                return False, "syntax error", False
            if not self.board.play(color, vertex, strict=False):
                return False, "illegal move", False
            self._write_sgf()
            return True, "", False
        if cmd == "genmove":
            if not args or args[0].lower() not in ("b", "w", "black", "white"):
                return False, "syntax error", False
            return True, self.genmove(args[0]), False
        if cmd in ("fixed_handicap", "place_free_handicap"):
            try:
                n = int(args[0])
            except (IndexError, ValueError):
                return False, "syntax error", False
            pts = self._fixed_handicap(n)
            if pts is None or not self._place_handicap(pts):
                return False, "invalid number of stones", False
            return True, " ".join(pts), False
        if cmd == "set_free_handicap":
            if not args or len(args) > BOARD_SIZE * BOARD_SIZE:
                return False, "invalid number of stones", False
            if not self._place_handicap([a.upper() for a in args]):
                return False, "bad vertex list", False
            return True, "", False
        if cmd == "undo":
            return (True, "", False) if self.board.undo() else (False, "cannot undo", False)
        if cmd == "showboard":
            return True, "\n" + self.board.render_ascii(), False
        if cmd in ("time_settings", "time_left", "kgs-time_settings"):
            return True, "", False
        if cmd == "llm-stats":
            return True, json.dumps(self.stats), False
        return False, "unknown command", False


def gtp_loop(engine: LLMEngine, stdin=sys.stdin, stdout=sys.stdout) -> None:
    for raw in stdin:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        ident = ""
        parts = line.split(None, 1)
        if parts[0].isdigit():
            ident = parts[0]
            line = parts[1] if len(parts) > 1 else ""
        ok, resp, quit_ = engine.handle(line)
        prefix = ("=" if ok else "?") + ident
        stdout.write(f"{prefix} {resp}\n\n" if resp else f"{prefix}\n\n")
        stdout.flush()
        if quit_:
            break


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="served model name (e.g. qwen3-8b-base, go-v4, go-v5c)")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8100/v1"))
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=8,
                    help="completions sampled per request; the first legal one is played")
    ap.add_argument("--max-attempts", type=int, default=2, help="requests per move before passing")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--eye-guard", choices=["off", "atari", "all"], default="atari",
                    help="refuse to fill an own true eye: 'atari' only when the fill leaves the "
                         "group with <=1 liberty (default), 'all' any true eye, 'off' never")
    ap.add_argument("--valid-list", choices=["legal", "empty"], default="legal",
                    help="'empty' = every empty point (exact training format); 'legal' = drop "
                         "suicide/ko points so the model is not invited to play them (default)")
    ap.add_argument("--chat", choices=["off", "main", "malkovich"], default="off",
                    help="post the model's reasoning for every move to the game chat via "
                         "stderr markers gtp2ogs understands ('main' = the opponent sees it, "
                         "'malkovich' = spectators/after the game only)")
    ap.add_argument("--chat-limit", type=int, default=CHAT_LIMIT,
                    help="max characters of reasoning per chat message (700 rarely truncates)")
    ap.add_argument("--log-dir", default=None, help="write per-move JSONL and per-game SGF here")
    ap.add_argument("--label", default=None, help="tag for log files (default: model name)")
    args = ap.parse_args(argv)

    label = args.label or args.model

    def log(msg: str) -> None:
        sys.stderr.write(f"[llm_gtp {label}] {msg}\n")
        sys.stderr.flush()

    client = OpenAIChatClient(args.base_url, args.model, args.temperature, args.top_p,
                              args.max_tokens, args.timeout, args.api_key, n=args.n_samples)
    engine = LLMEngine(client.complete, max_attempts=args.max_attempts,
                       eye_guard=args.eye_guard, valid_list=args.valid_list,
                       log_dir=args.log_dir, label=label, model_name=args.model, log=log,
                       chat=args.chat, chat_limit=args.chat_limit)
    log(f"ready: model={args.model} url={args.base_url} T={args.temperature} top_p={args.top_p} "
        f"n={args.n_samples}x{args.max_attempts} eye_guard={args.eye_guard} "
        f"valid_list={args.valid_list} chat={args.chat}")
    gtp_loop(engine)
    log(f"exit; stats={json.dumps(engine.stats)}")


if __name__ == "__main__":
    main()
