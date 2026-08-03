# Train with GRPO against KataGo evals, not SFT on KataGo's best move

KataGo can label every position with a single best move, which would make
supervised fine-tuning the obvious approach. We chose GRPO with a reward built
from KataGo's per-coordinate `policy`, `winrate` and `scoreLead` instead, because
SFT on the argmax throws away the shape of the evaluation — it cannot distinguish
a move that loses half a point from one that loses the game, and it gives the
model no signal at all about the reasoning it emits alongside the move. GRPO lets
the reward grade the whole completion (format, legality, move quality) and lets
near-optimal moves earn near-full credit.

## Consequences

The dataset must carry the full `katago_evals` map per position, not just a
target move, which is what makes it expensive to generate (~48 GPU-hours for
18,493 positions) and large on disk. It also means reward design is where most of
the project's iteration happens — see [0002](./0002-rank-based-reward-v4.md).
