# v4 reward: rank among legal moves, with weights adapted to root winrate

The v3 reward was a fixed composite, `0.5×policy + 0.3×score + 0.2×winrate`,
scoring each candidate by the absolute size of its winrate and score deltas. That
failed on the actual corpus: most positions are lopsided (root winrate above 0.95
or below 0.05), where every delta is compressed near zero and the reward becomes
noise. v4 replaces the score term with the chosen move's **percentile rank by
`scoreLead`** among the legal moves KataGo evaluated, and sets the relative
weight of the policy and rank terms from how close `root_winrate` is to 0.5.
Rank is scale-free, so it stays informative exactly where the absolute deltas
collapse.

## Considered options

- **Keep the v3 composite and just reweight it.** Rejected: no choice of weights
  fixes a signal whose dynamic range has collapsed.
- **Filter lopsided positions out entirely.** Rejected: it would discard most of
  an expensive corpus. v4 downsamples them to 30% instead, keeping some
  representation of decided positions without letting them dominate.

## Consequences

The v4 reward cannot be compared numerically against v1/v3 rewards — the scales
are different, so W&B reward curves are only meaningful within a generation.
