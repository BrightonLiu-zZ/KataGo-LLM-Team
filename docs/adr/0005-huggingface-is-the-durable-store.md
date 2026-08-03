# HuggingFace is the durable store; the server workspace is scratch

The workspace lives on a shared, quota-constrained server with no backups, and
`.gitignore` excludes everything large — datasets, checkpoints, merged models,
GGUFs. In April 2026 that space pressure forced a cleanup that freed ~150 GB by
pushing artifacts to HuggingFace and deleting the local copies. We are keeping
that arrangement deliberately: HuggingFace is the system of record for anything
too big for git, and the local workspace is treated as reproducible scratch.

The 2026-05-06 incident validated the principle and exposed the gap. Everything
that had been pushed to HuggingFace survived an accidental wipe of the entire
working tree; everything that had not — the raw KataGo analysis output, its two
downstream intermediates, and the `checkpoint-3000` adapter that produced the
published GGUF — was lost permanently. See
[RECOVERY-2026-05-06.md](../RECOVERY-2026-05-06.md).

## Consequences

An artifact is only safe once it is on HuggingFace or in git. Before deleting
anything large, confirm it exists in one of those two places — "it can be
regenerated" is not the same as "it is backed up" when regenerating costs 48
GPU-hours.
