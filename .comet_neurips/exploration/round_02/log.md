# Exploration log — round 2

baseline_sha: b2d2c48cd3eb

## Wins (visible to next improver, then folded into paper)

- Forward device/phase/gradient checks through the public API and CLI — device/phase/gradient check flags are now pinned by 25 tests covering API signatures, runtime behaviour, CLI flag presence, and end-to-end subprocess invocations with exit-code validation
- Mechanize the broadcast_add operator lemma in Lean to retire one operator-agnostic axiom — applyOpExt_sound_broadcast_add is mechanised sorry-free in Lean, retiring the operator-agnostic axiom for broadcast_add so only matmul remains under the agnostic witness

## Failures (silently reverted; never surfaced to reviewer)

- (none)
