---
name: Leaderboard submission
about: Submit or update a signed TensorGuard benchmark leaderboard entry
title: "[LEADERBOARD] "
labels: ["leaderboard", "benchmark"]
---

### Tool and release

Tool name, public URL, release/version, and exact command used to produce the
submitted verdict JSON.

### Corpus fingerprint

Paste the `fingerprint_sha256` from `reproducibility/leaderboard.json` used for
the run.

### Signed entry

Link to the PR/file under `benchmarks/leaderboard_entries/`, the
`signature.identity` principal, and the reviewed key in
`benchmarks/leaderboard_entries/allowed_signers`.

### Anti-overfitting attestation

Disclose any benchmark-specific tuning, manual triage, abstention rules, or
case-level debugging used before submission. If none, write "none".

### Reproduction notes

Environment, dependency versions, and any commands needed for maintainers to
re-run the tool on the frozen corpus.
