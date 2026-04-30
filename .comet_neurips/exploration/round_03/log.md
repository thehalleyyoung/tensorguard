# Exploration log — round 3

baseline_sha: bab54bbb37c9

## Wins (visible to next improver, then folded into paper)

- Wire check_devices/check_phases/check_gradients into verify_model and demonstrate flipped verdicts on a real-source example — forwarded check_devices/check_phases/check_gradients into verify_model with violation-level filtering and committed a three-entry JSON artifact where each flag flips the verdict from VERIFIED to REFUTED-PROOF on a real-source example
- Fix test_config_qkv_upgrade.py and add a third-party-mined gradient-flow validation corpus — fixed QKV test suite (already passing) and added mined gradient-flow corpus with 6 real-style snippets (detach/checkpoint/double-detach patterns) all REFUTED-PROOF by TensorGuard's gradient lattice

## Failures (silently reverted; never surfaced to reviewer)

- (none)
