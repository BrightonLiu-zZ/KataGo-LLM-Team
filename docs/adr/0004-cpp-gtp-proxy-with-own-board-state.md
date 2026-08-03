# The GTP proxy is C++ and maintains its own board state

Live play goes GUI → GTP proxy → LM Studio → model, and the proxy is a
self-contained C++ binary that tracks the 9×9 board itself, including flood-fill
capture resolution. The board tracking is not optional: GTP `play` commands
report only the stone placed, never the stones captured, so a proxy that merely
forwards moves cannot reconstruct the position — and the v4 model was trained on
prompts containing a full ASCII board plus the list of valid empty coordinates.
Getting that rendering wrong by even a character puts the model off-distribution.

C++ rather than Python (an earlier `my_gtp_proxy_1.0.py` exists) because the
target is a Windows desktop running the GUI and LM Studio; a single `.exe` with
no interpreter or dependency install is what makes the demo reproducible on a
machine we do not control.

## Consequences

The proxy duplicates Go rules that KataGo already implements, and the board
renderer and coordinate sort order must stay byte-identical to
`prepare_v4_dataset.py` and `train_grpo_v4.py`. A change to the training prompt
format is a change to `gtp_proxy_v4.cpp`.
