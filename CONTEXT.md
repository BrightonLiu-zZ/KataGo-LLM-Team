# KataGo-LLM Team

Teaching a large language model to play 9×9 Go by reinforcement learning against
a KataGo teacher, then shipping the result to a consumer GPU so it can be played
against in a normal Go GUI.

This file is a glossary and nothing else. Operational instructions live in
[CLAUDE.md](./CLAUDE.md); the record of past decisions lives in
[docs/adr/](./docs/adr/).

## Language

### The game

**Coordinate**:
A single intersection on the 9×9 board, written as a column letter `A`–`J`
(the letter `I` is skipped) followed by a row number `1`–`9`, row 1 at the
bottom. Example: `D4`. The one non-coordinate move is `PASS`.
_Avoid_: point, vertex, position, square

**Position**:
One board state from one real game, together with whose turn it is. A position
is the unit the model is asked to reason about and the unit a training record
describes.
_Avoid_: board, state, sample

**Valid empty coordinates**:
The list of coordinates that are unoccupied in a given position. It is rendered
into the prompt because the model is required to choose from it; a move outside
this list is illegal by construction.
_Avoid_: legal moves, candidate moves, available moves

### The teacher

**KataGo**:
The open-source superhuman Go engine used as the teacher. It is never trained
here — it is queried offline to label positions, and it is the reference the
reward function is derived from.

**KataGo evals** (`katago_evals`):
A map from coordinate to the three numbers KataGo reports for playing there:
`policy`, `winrate`, and `scoreLead`. `policy` is present for every evaluated
coordinate; `winrate` and `scoreLead` are present for roughly 78% of them.
_Avoid_: labels, annotations, scores

**Policy prior** (`policy`):
KataGo's raw prior probability of playing a given coordinate, in `[0, 1]`. It is
the densest signal in the dataset, which is why it survives in every reward
version.
_Avoid_: probability, confidence

**Score lead** (`scoreLead`):
KataGo's expected final point margin for the player to move, after playing a
given coordinate. Negative means losing points relative to the current position.
_Avoid_: score, margin, komi-adjusted score

**Root winrate** (`root_winrate`):
KataGo's winrate for the player to move in the position *before* any candidate
move is played. It measures how decided the game already is, and it is what the
v4 reward uses to decide how much to trust each signal.
_Avoid_: base winrate, position winrate

**Lopsided position**:
A position whose `root_winrate` is above 0.95 or below 0.05 — one side has
already effectively won. Such positions dominate the raw corpus and carry almost
no learning signal, so the v4 dataset downsamples them to 30%.
_Avoid_: decided position, blowout, saturated position

### The data pipeline

**Augmentation**:
Expanding one source position into 16 by applying the 8 symmetries of the square
(the D4 group) and swapping the two players' colours. Coordinates inside
`katago_evals` and inside the prompt are transformed with the board; the
`original_id` gains an `_aug<op>` and optional `_inv` suffix that identifies
which transform produced it.
_Avoid_: expansion, replication, symmetry sampling

**Identity record**:
The augmented record whose `original_id` ends in `_aug0` with no `_inv` — the
un-transformed one. Selecting these recovers the pre-augmentation dataset, which
is how `training_data_quality_filtered_v3.jsonl` was rebuilt after the incident.

**Quality cut**:
The filter that drops positions from the opening — anything with a move number
below 4, parsed from the `_pos<N>` suffix of `original_id`. It is the only
filtering step between the raw KataGo output and augmentation.
_Avoid_: cleaning, curation, quality filter

### Training

**Dataset generation** (v1 / v3 / v4):
A numbered iteration of the whole stack — dataset schema, reward function and
training script move together, so "v4" always means all three at once
(`training_data_v4.jsonl` + `train_grpo_v4.py` + the v4 reward). v2 exists only
as a checkpoint; there is no v2 dataset or script.
_Avoid_: version, revision, run

**Gate reward**:
The first, non-negotiable reward component: it checks that the completion
contains a parseable `MOVE:` and that the move names an empty intersection. It
returns graded partial credit (−1 no move, −0.5 illegal or occupied, 0 legal)
rather than a pass/fail, because a binary gate gives almost no gradient early in
training.
_Avoid_: format reward, validity check, legality reward

**Rank reward**:
The v4 core signal: the chosen move's percentile by `scoreLead` among the legal
moves KataGo actually evaluated. Being rank-based rather than difference-based
means it stays informative in positions where every winrate delta is tiny.
_Avoid_: score reward, percentile reward

**Adaptive weighting**:
The v4 scheme that sets the relative weight of the policy term and the rank term
from how close `root_winrate` is to 0.5. Balanced positions lean on the rank
signal; decided positions lean on the policy prior.

**Rollout log**:
The JSONL written by the non-scoring logging reward
(`src/model_training/logs/grpo_training_rollouts_v<N>.jsonl`), one line per
generated completion. It is the only record of what the model actually said
during training and is not reconstructable after the fact.
_Avoid_: trace, sample log, generation log

### Deployment

**Merged model**:
Base Qwen3-8B with a LoRA adapter folded into the weights, producing a standalone
full-precision checkpoint. It exists only as an intermediate on the way to GGUF,
and as the artifact published to HuggingFace.
_Avoid_: fine-tuned model, final model

**GTP proxy**:
The C++ binary that sits between a Go GUI and the language model. It speaks GTP
to the GUI, maintains its own 9×9 board (including capture resolution, which GTP
does not report), renders that board into the exact prompt format the model was
trained on, and translates the model's `MOVE:` back into a GTP response.
_Avoid_: adapter, bridge, wrapper, shim

**Reasoning channel**:
The GTP proxy's use of stderr to surface the model's `REASONING:` text to the
GUI's console, keeping stdout strictly to the GTP protocol. `<think>` blocks are
stripped before this is written.

**Thinking mode**:
Qwen3's native chain-of-thought behaviour, which must be disabled everywhere —
`enable_thinking=False` in training and Python inference, `/no_think` plus
`chat_template_kwargs` at the LM Studio API. Left on, it consumes the token
budget before the model ever emits `MOVE:`.
_Avoid_: reasoning mode, CoT
