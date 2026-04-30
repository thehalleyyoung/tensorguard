# Exploration log — round 6

baseline_sha: e553a7c83b4e

## Wins (visible to next improver, then folded into paper)

- Mechanised classifier for the 13 pen-and-paper handlers — AST-based mechanised classifier emits a 13-handler JSON certificate distinguishing T-Identity (shape-preserving/single-input) from T-Broadcast (multi-input broadcast) handlers, with a pytest suite and README audit target, replacing the silent over-count risk with a reproducible artefact.
- AST-based independent audit of the ≤12% tied/renamed-attribute prevalence — AST-level detector over all 488 real-source corpus blocks confirms zero tied/renamed-attribute parameter-sharing instances, yielding a Wilson-CI-bounded prevalence of ≤0.78% (95%) and a recomputed worst-case false-Verified deployment bound of 0.20%, replacing the unaudited ≤12% regex estimate.

## Failures (silently reverted; never surfaced to reviewer)

- (none)
