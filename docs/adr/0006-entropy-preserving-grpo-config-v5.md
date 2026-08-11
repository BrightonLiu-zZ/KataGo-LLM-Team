# exp05 trains with an entropy-preserving GRPO config and a reasoning-aware reward

Analysis of exp04 (W&B `fiery-firebrand-6`; `notebooks/exp01-04_analysis.ipynb`)
showed that its four visible problems are one mechanism. The reward is
continuous, so a group of 16 generations can only tie exactly when they all
choose the same move — and the fraction of zero-variance (zero-gradient) groups
climbed to 61% while token entropy fell 0.55 → 0.31 and completions shrank
90 → 43 tokens. Nothing opposed the sharpening: no KL anchor (`beta=0`),
symmetric clipping, per-group advantage scaling that amplifies low-variance
groups, and a reward that never reads the `REASONING:` text — so the reasoning
converged to move-irrelevant boilerplate ("decorative CoT", a documented
outcome of outcome-only RLVR). Separately, `warmup_ratio=0.02` of the
aspirational `max_steps=270000` put the *entire* 5,220-step run inside LR
warmup, and `temperature`, `scale_rewards`, `loss_type` and `beta` had been
running on silent TRL defaults since v1.

v5 (`train_grpo_v5.py`, exp05) therefore changes, together and from a fresh
base model:

- **Reward** — gate and core are unchanged; a third layer scores the reasoning
  (mention the chosen coordinate +0.05; shortness below 200 chars up to −0.1;
  high bigram overlap with the sibling generations of the same prompt up to
  −0.1). Total magnitude ≤0.2 so it cannot outweigh the KataGo signal; it also
  breaks the reward ties that produced the zero-variance groups.
- **Entropy** — `epsilon_high=0.28` (DAPO clip-higher), rollout
  `temperature=1.15` (ProRL), `scale_rewards="batch"` (Lite PPO), adaptive
  entropy regularization toward exp04's healthy early value (~0.5,
  Skywork-OR1; needs TRL ≥1.8).
- **Schedule** — `learning_rate=2e-6` constant with `warmup_steps=100`
  (absolute), `max_steps=10000` as a real horizon; exp04's own curve shows its
  learning happened below 2e-6.
- **Groups** — `num_generations=16` with `gradient_accumulation_steps=8`
  (128 completions, still 8 groups per step; ~78 GB peak on the 98 GB GPU).
- **Evaluation** — for the first time: 502 held-out positions split by source
  game (`prepare_v5_split.py`), evaluated every 500 steps, plus post-run GTP
  matches. Splitting row-wise would leak symmetry variants of eval positions
  into training, so whole games are held out.

## Considered options

- **Mechanism knobs only, reward untouched.** Cleaner attribution, but the
  zero-variance groups are caused by same-move ties, which sampling diversity
  alone cannot fully break, and reasoning would still receive no gradient.
- **The full ProRL recipe (KL `beta=0.001` + periodic reference reset,
  `epsilon_high=0.4`, temperature 1.2).** The strongest published anti-collapse
  recipe, but the most new machinery at once; deferred to exp06 if v5 still
  collapses.
- **Resuming from `checkpoint-5000`.** Rejected: that policy is already the
  collapsed state the config is designed to prevent, and the changed reward
  makes its curve incomparable anyway.
- **An LLM judge for reasoning quality.** Rejected: slow, unstable, and
  ungameable rule checks (coordinate mention, length, overlap) cover the
  observed failure modes.

## Consequences

- v5 reward values are on yet another scale; curves are comparable only within
  exp05 (extends ADR 0002).
- The reasoning terms are gameable in principle (e.g. padding text to 200
  chars); the rollout log (`grpo_training_rollouts_v5.jsonl`) must be spot-read
  during the run, which the destroyed-log incident makes newly possible.
- Training now depends on TRL ≥1.8, which forced the Dockerfile to a CUDA 12.9
  base for the Blackwell GPU (`sm_120`) under the host's 12.8 driver: cu130
  wheels cannot initialize, vLLM's bundled flash-attention ships PTX-only
  sm_120 kernels the 12.8 driver cannot JIT (so `train_grpo_v5.py` injects
  `attention_backend=FLASHINFER` into TRL's vLLM constructor), and flashinfer's
  sm_120 JIT requires nvcc ≥12.9 in the container. vLLM colocate is enabled
  with a documented one-line fallback (`USE_VLLM=False`), and
  `vllm_max_model_length=1280` caps the KV cache (measured prompts ≤518 tokens
  + 512 completion).
- Stopping rule: eval reward flat for 4 consecutive evals (2,000 steps) → stop
  manually; `save_total_limit=6` caps checkpoint disk use.

## Amendment (2026-08-03): what the reasoning layer can and cannot fix

Re-examining exp01 (whose v1 reward already contained a consistency + length
term) sharpened the expectations for the three reasoning terms:

- exp01's `thinking_quality_reward_func` was **never gamed** — no length
  padding, no coordinate-listing — but templated, move-irrelevant reasoning
  happened anyway. Its actual failure was the unclear core move reward
  (fixed since v4). Empirical conclusion: **a length window prevents length
  collapse (exp01 held length far better than exp04) but does not prevent
  boilerplate.**
- Therefore the **group-dedup term is the only new bet against boilerplate**.
  Its tie-breaking effect on zero-variance groups is mathematically certain;
  its effect on reasoning *quality* is the uncertain part. The consistency
  bonus (+0.05) is kept as a cheap tie-breaker despite its known loophole
  (listing coordinates); the loophole never materialized in exp01 and the
  rollout log is the monitoring line for it.
- **Success criteria for the reasoning layer:** ~~hard — run-averaged
  `frac_reward_zero_std` **< 15%** (exp04: ~29%)~~ **— retracted 2026-08-04,
  see below**; soft — spot-read rollouts show reasoning that references the
  actual position rather than templates.

### Correction (2026-08-04, exp05 already running at step ~920)

`frac_reward_zero_std` cannot serve as that criterion, because
`scale_rewards="batch"` changes what the metric measures. In TRL 1.9.2
(`grpo_trainer.py:2688–2708`) the flag is derived from whichever `std_rewards`
the scaling mode produced: under `"group"` (exp01–04) that is the per-group
std, so the metric is the true fraction of zero-gradient groups; under
`"batch"` (exp05) it is a single batch-wide std broadcast to every sample, so
the metric reads 0.000 unless all 128 completions in a step score identically.
exp05-vs-exp04 comparison on this number is meaningless.

The underlying mechanism is unaffected — advantages are
`(reward − group_mean) / batch_std`, so a degenerate group still contributes
zero gradient, and the dedup term still breaks those ties. Only the instrument
was lost. Replacement criteria, all already logged: `entropy` holding near the
0.5 target; `rewards/reasoning_quality_reward_func/mean` not sliding toward
−0.2 (the dedup term *is* a within-group similarity measure); and
`completions/mean_length` not collapsing. Detailed in `docs/WANDB-PANELS.md`.

For exp06: have `logging_reward_func` compute per-group reward std itself — it
receives the whole batch and can recover groups from runs of identical prompts,
exactly as `reasoning_quality_rewards` already does — which makes the metric
independent of `scale_rewards`. Not worth restarting exp05 to obtain.
- **Future work (exp06 candidate, deliberately not in exp05):** a
  KataGo-grounded precision term — the fraction of coordinates mentioned in
  the reasoning that fall in KataGo's top-k. The model cannot see the evals,
  so earning it requires genuine move identification, and precision (not
  count) makes coordinate-spraying self-defeating. Whether it is needed at
  all should be decided from exp05's rollout logs.

## Amendment 2 (2026-08-05): exp05 stopped at step ~1490 — root cause and exp05b

exp05 (W&B `tijdgbfj`) was stopped manually at step ~1490. Core reward sat at
the base-model level (~0.39) for the whole run — exp04 gained ~+0.04 over the
same span — while every protective mechanism worked as designed: entropy held
at 0.64 (target 0.5, controller never engaged: `entropy_coef` stayed 0.0),
completion length held at ~90 tokens, no length or entropy collapse. The
config cured exp04's disease and produced a policy that no longer learns.

**Discrimination experiment** (`tmp/discriminate_v5.py`, 2026-08-05), designed
to separate (a) vLLM-colocate weight-sync failure (rollouts from stale
weights) from (b) genuinely near-zero learning:

- `lora_B` Frobenius norm: 0.34 at step 500 → 0.41 at step 1000, largest
  element 8e-4. Updates are flowing but tiny, and growth is sublinear —
  a noisy, weakly-aligned gradient signal, not a dead optimizer.
- Rescoring 32 recent real vLLM rollouts (2,885 tokens) under base vs
  base+checkpoint-1000: mean per-token IS ratio **0.998**, mean |Δlogp|
  0.076. The trainer policy is behaviorally the base model, so there was no
  divergence for a broken sync to hide. **Weight sync is exonerated.**
- Greedy decode on 24 held-out eval positions: 11/24 move agreement, but
  identical move quality and identical templated text on both sides.

**Root cause: the effective step size is ~5× smaller than exp04's**, from two
compounding factors:

1. `scale_rewards="batch"` divides advantages by the batch-wide reward std
   (~0.4) instead of the per-group std — ~3× smaller advantage magnitudes
   than exp04's `"group"` scaling.
2. TRL 1.9.2's vLLM importance-sampling correction defaults to
   `vllm_importance_sampling_mode="sequence_mask"`, which multiplies each
   sequence's loss by exp(Σ per-token logp diff) (`grpo_trainer.py:3118`).
   The trainer's bf16 forward and vLLM's FlashInfer kernels disagree by only
   ~−0.006 logp per token — but over ~90-token completions the product is
   exp(−0.53) ≈ **0.59**, a flat ×0.59 tax on every gradient. This is the
   constant `sampling/importance_sampling_ratio/mean ≈ 0.59` seen on W&B.
   (Not a temperature mismatch: the trainer scales logits by the sampling
   temperature, `grpo_trainer.py:1519`. It is kernel-precision mismatch
   amplified by sequence-level exponentiation.)

Quantitative closure: exp04 gained ~+0.04 core per 1,000 steps; ÷5 ≈ +0.008,
which is exactly the drift measured in exp05's rollout log (0.396 → 0.404
over ~1,400 steps). Nothing is broken; the run was simply ~5× too slow to
show learning inside its own horizon.

**Decision — exp05b** changes exactly two knobs and nothing else, restarting
fresh from the base model (checkpoint-1000 embodies ~no learning and is not
worth resuming):

- `scale_rewards="group"` — restores exp04's advantage magnitude (~3×). The
  per-group difficulty bias it reintroduces is accepted; the dedup term
  already suppresses degenerate groups.
- `vllm_importance_sampling_mode="token_truncate"` — per-token ratios
  (~0.994) instead of the sequence product; the ×0.59 tax disappears while
  genuine per-token mismatch correction is retained (default clip max 3.0).

Side effect of returning to `"group"` scaling: `frac_reward_zero_std` is a
true per-group metric again, so the success criterion retracted in the
2026-08-04 correction above — run-averaged **< 15%** (exp04: ~29%) — is
**reinstated for exp05b**. Expected on W&B after the fix:
`importance_sampling_ratio/mean` ≈ 0.99 (it was ~0.59), and a visible core
reward slope within the first 1,000 steps (exp04 reference: +0.04).

## Amendment 3 (2026-08-06): exp05b also flatlined — the real bug was TRL's
## sleep-mode weight sync, now monkey-patched

exp05b ran ~2,000 steps and its reward panels were as flat as exp05's — but
the diagnostics were completely different, and they unmasked the true bug:

- `importance_sampling_ratio/mean` started at **1.00** (the token_truncate
  fix worked) and then decayed monotonically to **0.66**;
  `sampling_logp_difference` grew 0.41 → 3.41 nats/token across evals.
- `grad_norm` rose 0.16 → 0.99 and `‖lora_B‖_F` reached 1.05 at step 1500
  (2.5× exp05's at the same step) — the trainer was learning briskly.
- Rollout moves, text, and core reward stayed frozen at base-model level.

Trainer moving + generations frozen = **vLLM never received the updated
weights**. Root cause, confirmed in source: with
`vllm_enable_sleep_mode=True`, TRL 1.9.2 runs per step
`sync_weights()` (pushes merged trained weights) **then** `generate()`, and
generate() calls `collective_rpc("reload_weights")` — a workaround for vLLM
issue #29341 whose no-argument form **reloads the base model from disk**,
wiping the just-synced weights (`trl/generation/vllm_generation.py`, reload
at the top of generate; vLLM `gpu_model_runner.reload_weights`: "Use path of
original model if neither is provided"). Evals hit the same path without
even a preceding sync. Every rollout and every eval in exp05 and exp05b was
generated by the frozen base policy. This is upstream bug
[huggingface/trl#5312], fixed by PR #5313 (merged 2026-07-28) — **not in any
released TRL as of 1.9.2** (the latest release).

Why exp05 looked different: its ~5× effective-step-size deficit (Amendment
2) kept the trainer ≈ base, so the discrimination experiment correctly found
"trainer never moved" — the sync bug was present but unobservable. exp05b's
step-size fix made the trainer move, which exposed the sync bug as the next
(and final) bottleneck.

**Fix applied:** `train_grpo_v5.py` now monkey-patches the installed TRL,
mirroring PR #5313: the `reload_weights` RPC is neutered, and `generate()`
re-pushes trainer weights via `sync_weights()` whenever level-2 sleep has
discarded them since the last sync. Validated by a 2-step smoke run
initialized from exp05b's checkpoint-1500: core reward 0.41/0.73 per step
(base level is 0.39), entropy 0.47–0.49 (base: 0.64), rollout moves off the
base A1/D1 fixation, `importance_sampling_ratio/mean` 0.92–0.94 (was 0.66).
Generation provably uses trained weights now. Remove the patch when TRL
ships a release containing PR #5313.

Incidental finding from that smoke: checkpoint-1500's policy scores well
above base on its own rollouts — exp05b's trainer learned something real
despite sampling from the frozen base the whole time (off-policy updates
through truncated IS). Restarting fresh is still the right call: that
learning trajectory is biased and unquantified.

## Amendment 4 (2026-08-08): exp05c validates the config; crashes need a wrapper

exp05c (W&B `lc4hyp23`) is the first run of this generation whose reward curve
moves. Over 1,500 steps, with every knob from Amendments 2 and 3 in place:

| metric (250-step windows) | 0 | 250 | 500 | 750 | 1000 | 1250 |
|---|---|---|---|---|---|---|
| core reward | 0.401 | 0.455 | 0.469 | 0.483 | 0.492 | 0.491 |
| IS ratio | 1.000 | 0.999 | 0.994 | 0.993 | 0.992 | 0.992 |
| entropy | 0.640 | 0.552 | 0.534 | 0.514 | 0.521 | 0.518 |
| completion length | 91.1 | 90.6 | 87.3 | 87.8 | 88.4 | 87.8 |

Held-out eval (502 positions) core: 0.4853 → 0.5030 → 0.5059 at steps
500/1000/1500. The learning rate is ~+0.09 core per 1,500 steps, roughly 1.5×
exp04's +0.04 per 1,000 — while entropy holds at 0.52 (exp04 collapsed to
0.31) and length holds at 88 tokens (exp04: 43). **Anti-collapse and learning
are both online for the first time**, which retires the open question from the
original ADR: the entropy-preservation package does not cost learning signal.

One criterion misses: `frac_reward_zero_std` rose 0.5% → ~23% and flattened,
above the <15% target reinstated in Amendment 2 (exp04: ~29%). With entropy
and length stable this reads as the policy converging on good moves in easy
positions rather than degenerating, but it is the metric to watch; sustained
growth past ~30% should be treated as the exp04 failure mode returning.

**Operational finding.** exp05c died at step 1616 (after 19 h) with
`Fatal Python error: none_dealloc: deallocating None: bug likely caused by a
refcount error in a C extension` — a random refcount bug in one of the
container's C extensions (vLLM, FlashInfer, Triton and numba are all loaded),
unrelated to the training config and not fixable from here. At that rate the
remaining ~8,400 steps (~91 h) will hit it several more times. Two changes:

- `train_grpo_v5.py` accepts `RESUME_FROM=auto` (or a checkpoint path) and
  passes it to `trainer.train(resume_from_checkpoint=...)`. It previously
  called `trainer.train()` with no arguments and could not continue a crashed
  run at all.
- `src/model_training/run_with_restart.sh` wraps the trainer: restart from the
  newest checkpoint on abnormal exit, exit cleanly on completion or Ctrl-C,
  and abort after 3 failures shorter than `MIN_HEALTHY_SECONDS` so a broken
  config cannot hide inside a restart loop. Pass
  `WANDB_RUN_ID=<id> WANDB_RESUME=allow` to keep one continuous curve.

A crash costs up to `save_steps` (500 steps ≈ 5 h) of progress. Lowering
`save_steps` would shrink that but costs disk under `save_total_limit=6`;
left unchanged for now.

## Amendment 5 (2026-08-09): vLLM sleep mode disabled

exp05c crashed twice under the auto-restart wrapper, with two different
signatures:

| # | step | uptime | error |
|---|---|---|---|
| 1 | 1616 | 19 h | `Fatal Python error: none_dealloc` (C-extension refcount bug) |
| 2 | 1834 | 3 h 35 m | `CUDA Error: out of memory at cumem_allocator.cpp:163`, inside `sync_weights() → llm.wake_up(tags=["weights"])` |

The second one is explained by a number the no-sleep smoke exposed: the
training process's torch allocator holds **62.7 GiB allocated but 85.4 GiB
reserved** — a 22.7 GiB fragmentation gap. Sleep level 2 discards vLLM's
weights, so every `wake_up()` must re-map ~16 GiB of *physical* memory while
torch is sitting on nearly the whole card. TRL already calls `empty_cache()`
before waking, but that only returns unused cached blocks, not fragmented
ones. It is a matter of time before one wake fails.

**Decision: `vllm_enable_sleep_mode=False`.** vLLM then allocates its memory
once at engine init and never re-maps it, which deletes that failure mode
outright. It also makes the Amendment 3 monkey-patch inert — both the patch
and TRL's buggy `reload_weights` call live behind the same
`if self.mode == "colocate" and self.enable_sleep_mode:` guard — so
[trl#5312] can no longer bite either. The patch stays in the file for the day
someone re-enables sleep mode.

Validated by a 12-step smoke from the base model (2026-08-09):

| | sleep mode | no sleep |
|---|---|---|
| `importance_sampling_ratio/mean` | 0.992 | **0.9995–1.001** |
| step time | 38.4 s | **36.7 s** |
| steady-state GPU memory | 48.6 GB | **88.7 GB / 97.9 GB (90.6%)** |

Weight sync is *more* accurate without the sleep/wake cycle and each step is
~4% faster. The cost is the memory footprint: 88.7 GB is steady state, not a
spike, leaving ~9 GB of headroom.

**Accepted consequences.** This is a shared server, and 90.6% of the GPU is
effectively monopolizing it — coordinate before long runs, and expect an OOM
if a co-tenant claims more than ~9 GB. Two levers, in order, if that happens:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (targets the 22.7 GiB
fragmentation gap, no code change), then `vllm_gpu_memory_utilization` below
0.2 (tight: the 8B weights alone need ~16 GB of vLLM's ~19.6 GB budget).

**Also changed: `save_steps` 500 → 200.** At ~38 s/step a 500-step interval
banks progress every 5.3 h, but the observed crash interval was 3.6 h — so
crash #2 rolled the run back from step 1834 to 1500 and the next attempt was
on course to do the same. A run whose checkpoint interval exceeds its crash
interval can loop forever without ever saving, and the wrapper's crash-loop
guard cannot see it (each attempt runs for hours, far above
`MIN_HEALTHY_SECONDS`). 200 steps ≈ 2.1 h is comfortably under. Disk cost is
unchanged: `save_total_limit=6` × 512 MB ≈ 3 GB.

### Correction (2026-08-11): that `save_steps` change never took effect

exp05c's checkpoints stayed **500 apart for the rest of the run**.
`training_args.bin` in `checkpoint-6000` records `save_steps=200`, but
`trainer_state.json` records 500 — and the state is what counts.
`DefaultFlowCallback` decides when to save from `state.save_steps`
(`transformers/trainer_callback.py:586`), and resuming calls
`TrainerState.load_from_json()` (`trainer.py:1556`), which restores the value
baked into the checkpoint. `args.save_steps` is silently overridden, with no
warning.

Because exp05c resumed from `checkpoint-1500` — written while the setting was
still 500 — the fix was inert for the whole run. It cost nothing this time:
disabling sleep mode removed the crashes entirely (55 h, zero crashes, to step
6000). But the general rule matters and is now in the project docs:
**`save_steps`, `eval_steps` and `logging_steps` only take effect on a fresh
run.** To change one mid-generation, either start fresh or edit
`trainer_state.json` in the checkpoint being resumed from.
