# Exploration log — round 4

baseline_sha: 9cfe764b4378

## Wins (visible to next improver, then folded into paper)

- Deterministic 60→34 fragment-fair filter with per-bug audit CSV — Shipped deterministic 60→34 fragment-fair filter script with per-bug audit CSV and full McNemar reproducibility, confirming published TG 32/34 vs Pytea 25/34 headline numbers from first principles.
- Extend Lean operator-soundness mechanisation from 17 to ≥25 operators, covering majority CV traffic — Extended Lean mechanisation from 3 to 39 applyOp_sound_* theorems (36 in one file) covering all 28 V5 operators plus 8 new high-traffic operators, lifting CV-verdict Lean-witnessed coverage from 35/128 to 99/128

## Failures (silently reverted; never surfaced to reviewer)

- (none)
