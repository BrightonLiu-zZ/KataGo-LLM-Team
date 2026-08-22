# Measuring the model's strength in human amateur ranks (9×9)

Research digest and plan, written 2026-08-17. Question: **what amateur rank
(kyu/dan) does the LLM play at on 9×9?** — for base Qwen3-8B, v4
`checkpoint-5000`, and v5c `checkpoint-5000`. Everything the project has so
far is reward-model-internal (KataGo policy/rank agreement on held-out
positions); nothing has been played to the end against anyone.

Decisions taken with the user on 2026-08-17 are marked **[decided]**. Web
findings come from four parallel research passes (~400 page fetches); URLs are
the primary sources, "UNCERTAIN" marks single-source or unverifiable claims.

---

## 1. Facts that shape the design

1. **"Amateur rank" means three different things.** Report all three rather
   than one number:
   - *served rank* — a Glicko/Elo number earned against humans on a server;
   - *engine-ladder Elo* — result of matches against fixed-strength programs;
     precise and reproducible, but converting to a human rank needs anchors;
   - *performance rank* — inferred from move quality alone. **No 9×9
     calibration data or tool exists anywhere** (every published method —
     Moudřík 2015, Chen et al. ICLR 2025, Kuboki et al. 2025, KaTrain's
     `rank_utils.py` — is fitted on 19×19 KGS/Fox/OGS games).

2. **9×9 ranks are compressed.** One handicap stone ≈ 4–6 ranks; the standard
   9×9 handicap table drops komi **2 points per rank** of difference (Sensei's
   Library, "Handicap for smaller board sizes"; OGS redesign thread uses a 6×
   factor). Consequence: neighbouring ranks score only ~55–65% against each
   other, so distinguishing one rank by win-rate needs hundreds of games.
   Report **intervals**, and prefer komi-bisection (§3.B) over raw win-rate
   where an engine is the opponent.

3. **Servers.** KGS and IGS/Pandanet do not rate 9×9 at all. Fox (野狐) has an
   official engine adapter (FoxGTP) but requires a ≥3D account for AI
   certification, and 9×9 exists only in casual rooms that do not affect 段位.
   弈客 / 弈城 / 99围棋 have no bot API. **OGS is the only platform that is
   automatable *and* keeps a 9×9-specific human rating**:
   `https://online-go.com/termination-api/player/{id}` returns
   `ratings["9x9"]`, `ratings["live-9x9"]`, … each `{rating, deviation,
   volatility}`; `rank = ln(rating/525)·23.15` (0 = 30k, 20 = 10k, 30 = 1d).
   Since ~April 2026 human-vs-bot games no longer affect the *human*'s rating,
   but **bots are still rated from a sample of their human games**: at most one
   game per human per day counts, the game must reach scoring or the bot must
   resign, and the human's rating must be established (secondary sources:
   forums.online-go.com/t/59889, /t/60277 — UNCERTAIN on exact wording).
   9×9 is ~40% of all OGS games (27M-game sample). Bots may only *wait* for
   challenges (no automatch, ladders, tournaments) and must be registered bot
   accounts with a human admin. An LLM bot ("searchless transformer") was
   approved in July 2026 — precedent exists.

4. **KataGo's Human-SL model (`b18c384nbt-humanv0`) is not a 9×9 ruler.**
   lightvector: trained on 19×19 with "some" OGS/KGS other-size games mixed in
   (github.com/lightvector/KataGo/issues/1020); nobody has calibrated
   `rank_Xk` on 9×9 (issue #1184 unanswered). OGS live data for Human-SL bots
   on 9×9 is inconsistent (a "5k" profile bot rated 4d, a "5d" profile bot
   2.6k); even on 19×19 adjacent deep-kyu profiles are Elo-tied (PR #1209).
   Use it only as a stylistically human sparring partner, never as a label.

5. **CGOS 9×9 is alive** (`yss-aya.com:6809`, 5 min sudden death, komi 7.0,
   Tromp-Taylor scoring with no dead-stone removal, standings updated the day
   of research). Anchor `Gnugo-3.7.10-a1` = 1800; BayesElo table with 1,562
   programs (`yss-aya.com/cgos/9x9/bayes.html`). Useful points: random ≈ 540,
   AmiGo 1.8 ≈ 980, GnuGo 3.8 L10 ≈ 1946, Fuego/Pachi 10k playouts ≈
   2150–2350, KataGo raw policy (1 visit) ≈ 2600–2900, KataGo with search
   >3000 (above the Zen/CrazyStone that beat pros 6-0 in 2012–14).

6. **Nobody has assigned a kyu/dan rank to an LLM on 9×9.** The nearest work
   (LoGos, arXiv 2601.16447, Qwen2.5 + GRPO on 19×19) reports KataGo top-10
   hit rate and win-rate vs Human-SL profiles. Whatever we produce here is new.

7. **Model-specific unknowns for full games** (never observed — the model has
   only been scored on isolated positions):
   - the training prompt contains no komi and no rules;
   - "Valid empty coordinates" lists the model's own eyes, so it may fill them;
   - training positions all have ≥4 moves of history (none from the empty
     board), and passes appear in <2% of prompts — the opening and the
     endgame are off-distribution;
   - the model never learned to resign.
   The first local games must be watched for eye-filling and failure to pass;
   the GTP wrapper has an optional *eye guard* (refuse to fill an own true eye,
   resample, else pass). Any such guard must be declared in the report.

---

## 2. Anchors collected (OGS 9×9 ratings of public bots, queried 2026-08-16)

`rank = ln(rating/525)·23.15`. OGS bot ratings are noisy (they play whoever
challenges them, mostly beginners) — treat as ±2 ranks.

| Engine (OGS bot) | 9×9 rank | 19×19 rank |
|---|---|---|
| random-move-nixbot | rating floor (100) | – |
| Ami-Go-v1.8 | ~17k | ~17k |
| GnuGo L1 (2024 bot) / GnuGo L10 (OGS's own, 2014) / gnugo-nixbot 3.8 L10 | 16k / 15k / 11k | 15k / 15k / 15k |
| Fuego (OGS's own) / fuego-nixbot r2038 | 3.5k / 2k | 11k / 8k |
| pachibot / minipachi / LightPachi | 0.6k / 1d / 1d | 2.5k / 1.7k / 3.5d |
| kata-one-playout (KataGo 1.3.2, 1 visit) | 3.4d | 5.5d |
| 60b KataGo 1 playout / KataGo9x9_30b11 | 7.7d / 7.7d | 5.6d / – |
| katrain-18k/14k/10k/6k (calibrated-rank AI) | 8.5k / 5k / 2.8k / 1.2k | 11k / 8.7k / 4.1k / 0.7k |

Cross-server (2024 survey, ±1–2 stones): OGS 1d ≈ KGS 1d ≈ Fox 3k ≈ Tygem 4k
≈ EGF 2k ≈ AGA 1k; anoek's design target: OGS 1d ≈ EGF 1d − 0.7 stones.

---

## 3. Methods

### A. OGS 9×9 bots — the only automated path to a *served* human rank **[decided: build now]**

- One bot account per model: base Qwen3-8B, v4 `checkpoint-5000`, v5c
  `checkpoint-5000` **[decided: test all three]**. Rank *differences* between
  the three are more trustworthy than any absolute number.
- `gtp2ogs` 9.0.1 (Node ≥18; host has node 22): `allowed_board_sizes:[9]`,
  `allow_unranked:false`, `allow_ranked_handicap:false`,
  `allow_unranked_handicap:false`, live only (blitz disabled by default),
  `allowed_rank_range:["30k","9p"]`, `max_games_per_player:1`.
- Serving: one vLLM OpenAI server inside the `katago-llm` image
  (`--network host`), base weights loaded once + two LoRA adapters via
  `--enable-lora` (r=16, all projection layers — supported), so the GPU
  footprint is ~one 8B model, not three. **[decided: must be trivially
  pausable/resumable because the GPU is shared]** → a single control script
  `start | pause | resume | stop | status`; *pause* stops the container (frees
  the GPU) and the gtp2ogs processes.
- Read-out: `termination-api` `ratings["9x9"]` and its RD; stop when RD ≲ 60.
  Convert with the formula above; sanity-check against §2 anchors; translate
  to Fox/EGF/AGA with the survey table and quote ±2 stones.
- Everything below in §4 (pipeline) is this method.

### B. Local fixed-strength ladder + komi bisection — fastest, fully controlled

- Ladder (all with CGOS/OGS backing): random → AmiGo 1.8 → GnuGo 3.8 L1/L10
  (**anchor**: ~6–8k KGS, 11–15k OGS on 9×9) → Fuego 10k sims or Pachi
  `--nodcnn -t =10000` (~3k) → Fuego 100k / Pachi 40k (~1k–1d) → KataGo raw
  policy 1 visit (~3–5d) → KataGo 9×9 net 50–400 visits (superhuman).
- Runner: `gogui-twogtp` (maintained; Rémi Coulom's repo, 2025), `-referee`
  KataGo for scoring, ≥200 games per rung, colours alternated; the LLM side
  batches through vLLM so games run in parallel.
- **Komi bisection**: for each rung, adjust komi until the LLM scores 50%;
  Δrank = Δkomi / 2. This resolves sub-rank differences that win-rate at
  fixed komi cannot, and cross-checks anchors against each other.
- Output: win-rate-vs-rung curve (the "distribution over AI levels" the user
  asked for) + Bradley–Terry Elo pinned to the CGOS scale.

### C. Connect to CGOS 9×9 — zero opponent setup, public record

- Python client in `github.com/zakki/cgos` (`client-python/`), server
  `yss-aya.com:6809`, ~60 active bots incl. `net_1k…net_18k` (rank-named,
  identity unpublished — UNCERTAIN what they are).
- Caveats: 5 min sudden death per side (~45 moves → <5 s/move average — fine
  with vLLM); **Tromp-Taylor scoring without dead-stone removal** — the LLM
  will not capture dead stones and will lose won games at counting. Either
  accept the tax or add an endgame closer (declare it).

### D. Offline "performance rank" estimator — research-grade **[decided: documented, deferred]**

- Data: OGS rated 9×9 human games with per-size ratings
  (`/api/v1/players/{id}/games/?width=9&height=9&ranked=true`, throttle
  15–30 s; or za3k's bulk OGS dump), analysed with a fixed KataGo 9×9 net.
- Features per move: score loss, policy rank of the played move, top-k hit,
  Human-SL profile log-likelihoods. Fit rank ↔ features (Chen et al. ICLR
  2025 / Kuboki et al. 2025 report 80–92% exact-rank accuracy from 10–20
  games on 19×19; they also find *mean score loss alone cannot separate
  adjacent ranks*). Zero-training variant: lightvector's "guess the rank"
  (Σ log humanPolicy per profile, softmax) — only valid if Human-SL works on
  9×9, which must be checked on the OGS data first.
- Payoff beyond a rank number: a per-rollout performance-rank metric for
  exp06 monitoring. Effort: a small paper's worth.

### E. Human volunteers — small-sample, qualitative

- Never transcribe model moves into an online account (OGS forbids it for bot
  accounts; elsewhere it is cheating). Instead: 8–10 club players of known
  rank (15k–3d) play the model locally through Lizzie/KaTrain + GTP, 4 games
  each, colours alternated, komi bisection as in B. Value: what the model's
  mistakes look like to humans — nothing Elo can tell you.

### F. Rejected / grey

- GoQuest: 9×9-only human pool with a community rating→OGS table (GQ 2000 ≈
  OGS 2.8d) but only a reverse-engineered client, ToS unknown.
- AI Sensei "Ranked 9×9": AI opponents only, not automatable.
- Fox: 3D account needed, 9×9 not rated.
- KataGo Human-SL profiles as rank labels (see §1.4).

Order recommended: A now (this doc's §4); B as soon as the GTP wrapper
exists (it is the same wrapper); C opportunistically; D later; E if the club
is willing.

---

## 4. Method A pipeline (what is being built)

```
OGS ──ws──▶ gtp2ogs (host, node) ──GTP stdin/stdout──▶ src/rank_eval/llm_gtp.py
                                                            │ HTTP /v1/chat/completions
                                                            ▼
                                            vLLM OpenAI server (docker katago-llm,
                                            --network host, base + LoRA go-v4, go-v5c)
```

Components (all under `src/rank_eval/`, configs under `config/ogs/`):

| Piece | Role |
|---|---|
| `llm_gtp.py` | GTP engine: own 9×9 board (captures, suicide, simple ko; verified move-by-move against GnuGo), **last 8 moves** in history (training used `game_moves[-8:]`), passes rendered as `B pass`, prompt byte-identical to the v4/v5 training prompt (tested on the eval set), `enable_thinking=false`, 8 samples/request → first legal, `--valid-list legal`, `--eye-guard atari`, per-game JSONL + SGF log |
| `test_llm_gtp.py` | CPU unit tests (rules, prompt identity, parsing, GTP loop with a fake LLM) |
| `serve_llm.sh` | `start/stop/status/wait/logs` for the vLLM container `ogs-llm`: FlashInfer backend (sm_120 on the 12.8 driver, same as training), `--enable-lora --max-lora-rank 16`, `--max-model-len 1280`, `--gpu-memory-utilization 0.25` (≈24 GB); models `qwen3-8b-base`, `go-v4`, `go-v5c`; ~1.2 s per move |
| `ogs_bots.sh` | `start / status / drain / undrain / stop / render / ratings`; tmux session `ogs`, one window per bot; `stop` frees the GPU |
| `ogs_rating.py` | reads `termination-api` for player ids, prints 9×9 rating ± RD → rank |
| `twogtp.py` | local match runner (no Java): two GTP engines, GnuGo referee, optional per-side ending bot emulating gtp2ogs, parallel games, SGF + JSONL |
| `random_gtp.py` | uniform-random legal-move bot (ladder bottom rung) |
| `config/ogs/bot-common.json5` | gtp2ogs template (9×9 only, ranked only, no handicap, live+rapid, no correspondence, GnuGo ending bot in `pool` mode); rendered to `bot-{base,v4,v5c}.json5`; API keys in `config/ogs/keys.env` (gitignored) |
| `~/.local/bin/gnugo` | GNU Go 3.8 built from source (`CFLAGS=-fcommon` needed with GCC ≥10) — ending bot, referee, and ladder anchor |

Generation settings for play (documented so they can be revisited):
temperature 0.6, top_p 0.95, max_tokens 512, **8 samples per request, first
legal one played** (two requests max, then pass). Training sampled at T=1.15
to keep entropy up; play should be sharper, but greedy decoding would make
bot-vs-bot games deterministic. Sanity check of the serving path (200
held-out positions, v5 core reward, single sample): v5c 0.524 @T=1.15 /
0.513 @T=0.6 — matches the training eval (0.522), so vLLM + LoRA serve the
same policy the trainer measured. (v4 ckpt-5000: 0.577 / 0.546; base:
0.353 / 0.386.)

### 4.1 What the first local games showed (2026-08-17, `twogtp.py`, komi 7, area scoring)

The **position-level reward does not carry over to full games**. v5c
checkpoint-5000 vs GnuGo 3.8 level 10 (OGS 9×9 ≈ 14.6k): **0–18** over three
6-game sets, every loss a total wipe-out (W+88 / B+74). vs a uniform-random
legal-move bot (that only refuses to fill its own eyes): 4–2, then 2–4 with
the GnuGo ending bot attached — i.e. **roughly random level in full games**.
Board tracking was cross-checked move-by-move against GnuGo (126 moves, 0
mismatches), and the served policy reproduces the training eval reward, so
this is the model, not the harness. Failure modes, all visible in the SGFs
(`tmp/rank_eval/*/sgf`, per-move JSONL in `tmp/rank_eval/*/logs`):

1. **It does not know the game ends.** It never passes on its own (training
   positions come from mid-game; passes are <2% of prompts) and after the
   board is settled it keeps playing *inside its own territory*, filling its
   own eyes until the group dies. Fix in play: gtp2ogs' `ending_bot` (GnuGo)
   decides when to pass/resign — but GnuGo will not vote "pass" while it
   still sees moves, so much of the self-destruction happens before that.
2. **It proposes suicide moves.** With the exact training prompt (every empty
   point listed as "valid"), 30% of its moves vs GnuGo ended in a forced pass
   because all 16 samples were suicides (the training gate only checked
   emptiness). `--valid-list legal` (drop suicide/ko points from the list)
   brings first-sample legality from 60% to 89% and forced passes to 3%.
3. **It fills its own eyes even when that is the killing move.** Blocking
   *every* own-eye fill (`--eye-guard all`) was worse: it then insists on
   the same point in all 16 samples and passes instead. Default is
   `--eye-guard atari`: block only a fill that leaves the group with ≤1
   liberty.
4. Base Qwen3-8B, v4 and v5c all lose 0–6 to GnuGo L10; the rank card will
   mostly separate them from each other, not from GnuGo.

Consequences for the plan: (a) the OGS rank will very likely land at
**25–30k** — that is the honest number, and it is the point of running the
bots; (b) for exp06 the reward needs a game-level component (pass/terminal
awareness, life-and-death), because "good move on a mid-game position" is
learned while "don't kill your own group" is not; (c) all guards are minimal
and declared: legal-only valid list, atari-only eye guard, GnuGo pass/resign
votes after 35% of the board (28 moves).

### 4.2 Launch procedure

Human steps, once per bot (base / v4 / v5c): create the OGS account with a
"botty" name (e.g. `qwen3go-v5c`) → post in the OGS forum *API Development*
category asking for the bot flag, naming your human account as admin (recent
requests were approved within hours) → once flagged, open the bot's profile
while logged in as the human account and generate the API key → put
username / key / numeric id into `config/ogs/keys.env` (template:
`keys.env.example`).

```bash
bash src/rank_eval/ogs_bots.sh start    # vLLM (docker "ogs-llm", ~24 GB) + 3 gtp2ogs in tmux "ogs"
bash src/rank_eval/ogs_bots.sh status   # server, tmux windows, active games per bot, GPU
bash src/rank_eval/ogs_bots.sh ratings  # 9x9 rating ± RD → rank, per bot
python src/rank_eval/ogs_feedback.py    # what the humans said in the game chat (see 4.4)
bash src/rank_eval/ogs_bots.sh drain    # co-tenant needs the GPU: decline new games, finish current
bash src/rank_eval/ogs_bots.sh stop     # ...then free the GPU (or immediately, forfeiting ~1 game/bot)
bash src/rank_eval/ogs_bots.sh start    # resume later; ratings persist on OGS
```

Local matches (Method B, same engine, no OGS needed):

```bash
python src/rank_eval/twogtp.py \
  --black "python src/rank_eval/llm_gtp.py --model go-v5c --log-dir tmp/rank_eval/logs --label v5c" --black-name v5c \
  --white "$HOME/.local/bin/gnugo --mode gtp --level 10 --chinese-rules --capture-all-dead" --white-name gnugo10 \
  --black-ending "$HOME/.local/bin/gnugo --mode gtp --level 10 --capture-all-dead" \
  --games 20 --alternate --komi 7 --jobs 6 --sgf-dir tmp/rank_eval/sgf --results tmp/rank_eval/res.jsonl
python src/rank_eval/test_llm_gtp.py      # CPU unit tests incl. prompt byte-identity vs eval set
```

Report format ("rank card", one per model): OGS 9×9 rating ± RD and rank,
number of counted games, date; plus — when B/C are run — CGOS Elo, ladder
win-rate curve, komi-equivalents vs anchors. Always state the guard/decoding
settings used.

### 4.3 First 14 h live: zero games, and why (2026-08-18)

The three bots ran from 2026-08-18 00:51 to 14:25 with **0 games**. Nothing
was broken in the pipeline — all of it was verified live: gtp2ogs
authenticated, the GnuGo ending-bot pool was ready, the bots appeared in the
server's `active-bots` broadcast (the list the "Play → Computer" panel is
built from), and the config file hot-reload worked. **Our own config
rejected, or hid us from, every challenger.** Evidence and mechanism:

* **5 real challenges arrived and were declined by us** (`main.ts:318` in
  `logs/ogs/console-*.log`): 2× `unranked_not_allowed`, 2×
  `komi_out_of_range`, 1× `board_size_not_allowed` (19×19). One player
  (`manro`) retried three different settings within 80 s and gave up.
* **Demand exists.** In one 4-hour window `amybot-beginner` finished 54 ranked
  9×9 games and `Agapanthus` 62 — 9×9 human-vs-bot traffic is dozens of games
  per hour. Of the 36 bots online, ours were the only three that refused
  unranked games *and* the only three restricted to a single board size.
* **The OGS play panel greyed us out by default.** `src/lib/bots.ts`
  (`getAcceptableTimeSetting`) disables a bot when the requested settings
  don't fit. The panel's defaults are 9×9 / rapid / fischer — good for us —
  but it always probes with `handicap: true` unless the user sets
  "Handicap: disabled", and `allow_ranked_handicap: false` therefore disabled
  our bots for everyone on default settings ("Bot doesn't accept ranked games
  with handicap"). 25 of 36 online bots allow ranked handicap.
* **Unranked is the default for bot games.** The panel creates the challenge
  with `ranked = preferences["automatch.bot-ranked"]`, whose default is
  `false`, so the standard flow sends an *unranked* challenge — which
  `allow_unranked: false` declined. (The same file has an inverted check for
  the unranked case, `!options.ranked && bot.config.allow_unranked`, so the
  UI's grey-out never protects against this either way.)
* **`komi_out_of_range` fires on automatic komi.** gtp2ogs compares
  `komi < allowed_komi_range[0]`; automatic komi arrives as `null` (and
  handicap games use 0.5), and `null < 5.5` is true. The default range is
  `[-99, 99]`; ours was `[5.5, 7.5]`.

Changes made (`config/ogs/bot-common.json5`): `allow_unranked: true`,
`allow_ranked_handicap` / `allow_unranked_handicap: true`,
`allowed_komi_range: [-99, 99]`, `concurrent_games` 4 → 2 per speed (one vLLM
server behind three bots, 8 samples per move). Blitz stays off: the 9×9 blitz
preset is 30 s + 5 s, too tight for ~1–3 s/move sampling under load.
`llm_gtp.py` gained `fixed_handicap` / `place_free_handicap` /
`set_free_handicap`, without which the now-allowed handicap games would break.

Consequences to keep in mind when reading the numbers:

* Unranked games do **not** move the OGS 9×9 rating. They still produce SGFs
  and opponent ranks, so the offline estimate (Method D) can use them; the
  served rank will accumulate more slowly than the game count suggests.
* Handicap games are rated by OGS but shift the reading; record handicap and
  komi per game and report even-game results separately.

**Operational gotcha:** the server only re-publishes a bot's config to
`active-bots` on a visibility change, not on every hot-reload — after editing
the config run `ogs_bots.sh drain` then `undrain` (or restart) and verify the
new values are in the broadcast before assuming they are live.

### 4.4 Talking to the opponent: per-move reasoning + feedback collection (2026-08-19)

The bots now say what they are thinking, and ask the human what they thought.
Both directions run through the OGS game chat; no extra credentials are
involved (the bot API key is enough — the account's verified email only
matters for OGS's own chat permissions, which are a property of the account,
not of this pipeline).

* **Outgoing, one message per move.** gtp2ogs scans the engine's *stderr* with
  `/(DISCUSSION|MALKOVICH|MAIN):(.*)/` when `bot.send_chats` is on and posts
  the capture to the game chat. `llm_gtp.py --chat main` therefore prints
  `MAIN: <move>: <the model's REASONING text>` after every `genmove`
  (`--chat malkovich` would hide it from the opponent until the game ends;
  `off` is the default, so local ladder games stay silent). The text is
  sanitized: collapsed to one line, truncated to `--chat-limit` (400) chars,
  and any further `MAIN:`/`MALKOVICH:`/`DISCUSSION:` marker inside the model's
  own output is defanged so a completion cannot forge a second chat or switch
  channel. Rejected samples (illegal move, eye guard) are reported as
  `[2 rejected samples]`, and a forced pass says so — the human sees the same
  guard events the logs record.
* **Incoming, at the end.** `farewell` (en / zh-cn / zh-tw) asks for the three
  things worth knowing: how strong it felt in kyu, its worst move, what it
  should learn next. `log_game_chat: true` makes gtp2ogs log every message the
  *opponent* sends, and gtp2ogs also reconnects to a finished game when a
  `lateChatReceivedInGame` notification arrives — so replies typed *after* the
  game still land in the logfile.
* **Collection.** `python src/rank_eval/ogs_feedback.py` parses those log lines
  into one record per message (`ts`, `bot`, `game_id`, `user`, `body`, game
  URL) plus the game result, de-duplicating the `console-*.log` / `gtp2ogs-*.log`
  copies; `--json`/`--out` write JSONL, the default prints a digest grouped by
  game.

Tests: `python src/rank_eval/test_llm_gtp.py` covers the chat format against
gtp2ogs' own regex, the sanitizer (including a completion that tries to forge
a `MALKOVICH:` line), one-chat-per-move over a scripted game, the forced-pass
message, and the feedback parser. `node src/rank_eval/chat_probe.js
config/ogs/bot-v5c.json5 6` is the integration check: it reads the *rendered*
config, spawns the engine exactly as gtp2ogs does, applies the regex **read
out of the installed gtp2ogs bundle** (so an upgrade that changes the marker
fails loudly instead of silencing the bots), and reports the chat that each
move would have produced. It needs the vLLM server running.

---

## 5. Source index (primary URLs)

- OGS: gtp2ogs https://github.com/online-go/gtp2ogs ; bot policy
  https://github.com/online-go/online-go.com/wiki/Running-Bots-On-OGS ;
  rating formula https://forums.online-go.com/t/2021-rating-and-rank-adjustments/33389 ;
  bot-rating change 2026 https://forums.online-go.com/t/games-with-bots-are-annulled/59889 ;
  rank survey 2024 https://forums.online-go.com/t/go-rankings-survey-feb-2024/50803
- KGS rates 19×19 only http://www.gokgs.com/help/rank.html ; Fox AI adapter
  https://www.foxwq.com/soft/aiprogramandmanual.html
- CGOS http://www.yss-aya.com/cgos/ ; client https://github.com/zakki/cgos ;
  9×9 BayesElo http://www.yss-aya.com/cgos/9x9/bayes.html
- Human-SL: docs https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md ;
  board-size answer https://github.com/lightvector/KataGo/issues/1020 ;
  ladder PR https://github.com/lightvector/KataGo/pull/1209
- Handicap on small boards https://senseis.xmp.net/?HandicapForSmallerBoardSizes ;
  https://forums.online-go.com/t/proposal-redesign-small-board-komi-and-handicap/50314
- Rank estimation: Chen et al. https://arxiv.org/abs/2502.17109 ; Kuboki et
  al. https://arxiv.org/abs/2505.00279 ; Moudřík https://arxiv.org/abs/1512.08969 ;
  KaTrain rank_utils https://github.com/sanderland/katrain-bots
- LLM Go prior work: LoGos https://arxiv.org/abs/2601.16447
- Engines: Pachi https://github.com/pasky/pachi ; GnuGo https://senseis.xmp.net/?GNUGo ;
  GoAIRatings https://github.com/breakwa11/GoAIRatings ; gogui-twogtp
  https://github.com/Remi-Coulom/gogui
