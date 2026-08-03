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
- Training now depends on TRL ≥1.8, which forced the Dockerfile to a CUDA 12.8
  base for the Blackwell GPU (`sm_120`); vLLM colocate is enabled with a
  documented one-line fallback (`USE_VLLM=False`).
- Stopping rule: eval reward flat for 4 consecutive evals (2,000 steps) → stop
  manually; `save_total_limit=6` caps checkpoint disk use.
