# Reading the W&B panels for exp05

Every metric the exp05 run writes to
`brighton_zz-uc-san-diego/huggingface` (run `exp05-grpo-v5`): how it is
computed, what it actually means, and which ones decide anything.

Definitions were read out of the installed TRL 1.9.2 source
(`trl/trainer/grpo_trainer.py`), not from memory; the example values come from
the 2026-08-04 smoke test. Config values referenced here live in
`config/exp05.yaml`; the reasoning behind them is ADR 0006.

---

## 1. The six panels that actually decide things

Everything else is diagnostic. These six answer "is the run healthy, and when do
I stop it?"

| Panel | Healthy | Meaning of a bad reading |
|---|---|---|
| `entropy` | ~0.4–0.7, no sustained decline | **The exp05 headline metric.** Falling = the policy is narrowing = the exp04 failure repeating. exp04 went 0.55 → 0.31. |
| `completions/mean_length` | ~80–100 tokens, stable | The visible twin of entropy collapse. exp04 went 90 → 43 tokens. Correlation between the two in exp04 was 0.93. |
| `eval_reward` | rising, then plateau | **The stopping rule**: flat for 4 consecutive evals (2,000 steps) → stop manually. First generation to have a held-out eval at all. |
| `rewards/katago_composite_reward_func/mean` | rising toward 1.0 | The actual Go-strength signal. If it is flat while `reward` rises, the model is gaining from the cheap layers, not from playing better. |
| `rewards/reasoning_quality_reward_func/mean` | near +0.05, not drifting to −0.2 | Reasoning is getting short and/or homogeneous. See §2 — this is now the best within-group diversity proxy available. |
| `grad_norm` | steady, ~0.1–0.3 | Rising trend = instability (exp04 drifted 0.2 → 0.7 at lr 5e-6, one reason exp05 uses 2e-6). Spikes = something is wrong. |

---

## 2. Caveat: `frac_reward_zero_std` does NOT mean in exp05 what it meant in exp04

This was going to be the hard success criterion ("run-averaged < 15%, versus
exp04's ~29%"). **That comparison is invalid and the criterion is
unmeasurable as written.** Found while writing this doc, at step ~920.

The metric is `is_std_zero.float().mean()`, and `is_std_zero` comes from
whatever `std_rewards` the configured `scale_rewards` mode produced
(`grpo_trainer.py:2688–2708`):

- `scale_rewards="group"` (exp01–04): `std_rewards` = std **within each group of
  generations**, so the metric = *fraction of groups whose completions all
  scored identically* → the real zero-gradient rate. exp04: ~29% average, 61%
  peak.
- `scale_rewards="batch"` (**exp05**): `std_rewards` = one **batch-wide** std
  broadcast to every sample. `is_std_zero` is then all-or-nothing for the whole
  batch of 128, so the metric reads **0.000 at every step** unless all 128
  completions score exactly the same. The smoke test showed exactly this.

**What is still true:** advantages are `(reward − group_mean) / batch_std`, so a
group whose completions all score identically *still* contributes zero gradient.
The waste mechanism is unchanged — only the instrument that measured it is gone.
The dedup reward term that breaks those ties is also unchanged and still working.

**Replacement criteria (use these instead):**

1. `entropy` staying near the 0.5 target — the upstream cause of group
   degeneracy, and the thing exp05's whole config exists to defend.
2. `rewards/reasoning_quality_reward_func/mean` not sliding toward −0.2 — the
   dedup penalty is *by construction* a within-group similarity measure, so its
   value is a direct read on whether siblings are converging on the same text.
3. `completions/mean_length` holding — the other half of the collapse signature.
4. `reward_std` — batch-wide spread; weak, because it mixes "prompts differ in
   difficulty" with "siblings differ from each other", but a collapse to near
   zero is still meaningful.

**To get the real metric back in exp06:** have `logging_reward_func` compute the
per-group std of total reward itself (it already receives the whole batch and
the group structure is recoverable from runs of identical prompts, exactly as
`reasoning_quality_rewards` does it) and log that. Cheap, CPU-only, and
independent of `scale_rewards`. Not worth restarting exp05 for — that would cost
the run's progress to regain observability of a mechanism the other four signals
already cover.

---

## 3. Full panel reference

### 3.1 Rewards

The four reward functions are summed into one number per completion. All
per-function panels are means over the 128 completions in the batch.

| Panel | Computation | Meaning |
|---|---|---|
| `reward` | mean of (gate + core + reasoning + logging) over the batch | The total signal being optimized. Smoke: ~0.32–0.42. Not comparable to exp01–04 — every generation redefines the scale (ADR 0002). |
| `reward_std` | `nanstd(rewards)` over **all** completions in the batch (source line 2749) — *not* per-group | Overall spread. Mixes across-prompt difficulty with within-prompt diversity. |
| `rewards/format_and_legality_reward_func/mean` | gate layer: −1 no `MOVE:` / −0.5 illegal or occupied / 0 legal | Format+legality health. Should climb to 0 and stay. exp04 reached a flat 0 around step 2,070 — that panel going dead is a *success*, not a fault. Smoke: −0.066 (a few % still illegal). |
| `rewards/format_and_legality_reward_func/std` | std of the gate values | Falls to 0 once the gate is fully learned. |
| `rewards/katago_composite_reward_func/mean` | `w_policy × R_policy_log + w_rank × R_rank`, weights set by how close `root_winrate` is to 0.5 | **Actual Go strength**, 0 (worst) to 1 (best). Smoke: ~0.35–0.44. |
| `rewards/katago_composite_reward_func/std` | std of the above | Near-zero = the model plays uniformly (either everywhere-good, or collapsed). |
| `rewards/reasoning_quality_reward_func/mean` | coordinate mention (+0.05) + length window (0 to −0.1) + group dedup (0 to −0.1); range [−0.2, +0.05] | Smoke: +0.043, i.e. most completions earn the mention bonus and pay almost nothing. Drifting negative = reasoning shrinking below 200 chars and/or siblings converging on the same wording. |
| `rewards/reasoning_quality_reward_func/std` | std of the above | — |
| `rewards/logging_reward_func/mean`, `/std` | always exactly 0.0 | Non-scoring. It exists only to write the rollout log. A nonzero value means something is broken. |
| `frac_reward_zero_std` | `isclose(std_rewards, 0)` averaged | **See §2 — reads 0.000 in exp05 by construction.** |

### 3.2 Completions

| Panel | Computation | Meaning |
|---|---|---|
| `completions/mean_length` | mean generated tokens per completion | The collapse canary. exp04: 90 → 43. |
| `completions/min_length`, `max_length` | batch extremes | A shrinking `max_length` is an early collapse warning. |
| `completions/clipped_ratio` | fraction that hit `max_completion_length=512` without emitting EOS | Must stay ~0. exp02 sat at ~1.0 — every completion was truncated before `MOVE:` and the run received no gradient at all. |
| `completions/mean_terminated_length`, `min_`, `max_` | same stats over only the completions that ended naturally | Identical to the plain versions whenever `clipped_ratio` is 0. |

### 3.3 Training dynamics

| Panel | Computation | Meaning |
|---|---|---|
| `loss` | DAPO surrogate + entropy regularization | **Nearly meaningless in absolute terms.** Advantages are normalized to mean 0, so this hovers around 0 and is often negative. Do not read it as "how good the model is". |
| `policy_loss` | the surrogate term alone, without the entropy part | Same caveat. |
| `grad_norm` | L2 norm of gradients before clipping (`max_grad_norm=1.0`) | Stability gauge. See §1. |
| `learning_rate` | current LR from the scheduler | Should ramp 0 → 2e-6 over the first 100 steps, then hold flat. **In exp04 this never reached plateau** — `warmup_ratio=0.02 × max_steps=270,000` = 5,400 warmup steps for a 5,220-step run. Seeing it flat at 2e-6 confirms the bug is fixed. |
| `entropy` | masked mean per-token entropy of the policy over generated tokens | See §1. Smoke: ~0.65. |
| `entropy_coef` | adaptive weight, moved by ±0.005 per step toward `entropy_target=0.5`, clipped to [0, 1] | **Tells you which way the thermostat is pushing.** Entropy above target → decays toward 0 (smoke did exactly this: 0.01 → 0). Entropy below target → rises to push exploration back up. A coefficient pinned high for a long stretch means the config is fighting a collapse and losing. |
| `clip_ratio/low_mean` | fraction of tokens clipped at the lower bound (1 − ε = 0.80) | Updates that wanted to suppress a token hard. |
| `clip_ratio/high_mean` | fraction clipped at the upper bound (1 + ε_high = **1.28**) | The asymmetry is the point: DAPO clip-higher raises this ceiling so rare-but-good exploratory tokens can actually be reinforced. Compare against `low_mean` to see whether it is doing anything. |
| `clip_ratio/region_mean` | total clipped fraction | Smoke: ~1e-4 to 1e-3. Above a few percent = the policy is moving too far per step. |
| `clip_ratio/low_min`, `high_max` | extremes across the gathered batch | Diagnostic only. |

### 3.4 vLLM ↔ trainer consistency

New in exp05 — exp01–04 generated in-process, so these panels did not exist.
Generation runs in vLLM while gradients come from the HF forward pass; the two
use different kernels, so their log-probabilities differ slightly and TRL
corrects for it (`vllm_importance_sampling_correction=True`, cap 3.0).

| Panel | Computation | Meaning |
|---|---|---|
| `sampling/sampling_logp_difference/mean`, `/max` | \|log p(vLLM) − log p(trainer)\| for the same tokens | Small is normal (smoke: mean ~0.025). A **growing** gap means the two engines are drifting apart — e.g. LoRA weights not syncing into vLLM correctly. |
| `sampling/importance_sampling_ratio/mean`, `/min`, `/max` | `exp(logp_trainer − logp_vllm)`, clipped at 3.0 | Should sit near 1.0 (smoke: 0.97–1.02, max ~2.5–3.0). Persistent drift away from 1.0 means the correction is working hard, i.e. generation no longer matches training. |

### 3.5 Progress and performance

| Panel | Meaning |
|---|---|
| `step_time` | Seconds per optimizer step (128 completions). Measured steady state: **44 s/step**. |
| `epoch` | Fraction of the 105,596-row training set consumed. 10,000 steps × 8 prompts = 80,000 prompts ≈ **0.76 epoch** — exp05 will not finish a single pass over the data. |
| `num_tokens` | Cumulative tokens processed (prompts + completions). |
| `profiling/Time taken: GRPOTrainer.compute_loss` | Internal profiling, seconds. Diagnostic only. |
| `train/global_step`, `_step`, `_runtime`, `_timestamp` | W&B bookkeeping. `_runtime` is wall-clock seconds since the run started. |

### 3.6 The `eval_` twins (every 500 steps, 502 held-out positions)

Every panel above has an `eval_`-prefixed version, computed the same way but on
the held-out split — positions from **source games that never appear in
training**, so symmetry variants cannot leak (`prepare_v5_split.py`).

| Panel | Note |
|---|---|
| `eval_reward` | **The stopping-rule number.** Flat across 4 consecutive evals = stop. |
| `eval_runtime` | Seconds for the full eval. Smoke: 262 s for 32 positions → ~68 min projected for 502, so each eval costs a bit over an hour. |
| `eval_samples_per_second`, `eval_steps_per_second` | Throughput. Low by design — every eval position is fully generated. |
| `eval_loss` | Same caveat as `loss`: not a quality measure. |

Eval still samples at `temperature=1.15`, so individual eval points are noisy.
**Read the trend across evals, never a single point.**

### 3.7 The System tab (W&B automatic)

GPU utilization, GPU memory allocated, temperature, power draw, CPU, disk,
network. Two uses here: confirm the ~78 GB peak matches the estimate, and
confirm the GPU is not idling between steps (which would point at a data-loading
or vLLM-wakeup bottleneck rather than a training problem).

---

## 4. What known failure modes look like

Recognizing these from the curves alone, based on exp01–04
(`notebooks/exp01-04_analysis.ipynb`):

**Entropy collapse (exp04, the thing exp05 is built to prevent).**
`entropy` and `completions/mean_length` fall together, monotonically.
`rewards/reasoning_quality_reward_func/mean` slides toward −0.2 as siblings
converge. Under exp04's `scale_rewards="group"` this also showed as
`frac_reward_zero_std` climbing to 61% — in exp05 that panel will stay at 0
regardless (§2), so watch the first three.

**Truncation death (exp02).** `completions/clipped_ratio` ≈ 1.0, gate reward
pinned at −1.0, `grad_norm` = 0. The model never emits `MOVE:` because thinking
mode ate the token budget. Prevented in v5 by `enable_thinking=False`, but the
signature is worth knowing: **`grad_norm` at exactly 0 means the run is dead**,
whatever else the curves say.

**Never leaving warmup (exp02–04).** `learning_rate` rising forever, never
plateauing. Fixed in exp05 by `warmup_steps=100`; verify it flattens at 2e-6
early and stays there.

**Reward going up for the wrong reason.** `reward` rises while
`rewards/katago_composite_reward_func/mean` is flat — the gains are coming from
the gate or the reasoning layer, not from better moves. Cross-check by reading
`src/model_training/logs/grpo_training_rollouts_v5.jsonl`, which records the
full per-sample reward breakdown.

**Reward hacking of the reasoning layer** (gameable in principle, never yet
observed). Two known loopholes: listing many coordinates so the
coordinate-mention bonus is always earned, and rotating templates to dodge the
dedup penalty. Neither shows up in any panel — the rollout log is the only place
they are visible. Spot-read it periodically.
