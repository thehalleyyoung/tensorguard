● Read brainstorm.prompt.md
  │ .comet_neurips/exploration/round_02/brainstorm.prompt.md
  └ 112 lines read

```
### Candidate 1: Forward device/phase/gradient checks through the public API and CLI
goal: Eliminate the reviewer obligation that `check_devices`, `check_phases`, and `check_gradients` are advertised as features but not exposed by the public API/CLI. This directly raises Contribution by ~0.5 (multi-feature system becomes a real user-facing artifact) and Presentation by ~0.5 (README/paper architecture matches the shipped artifact).
plan:
  - Locate the public entry points (likely under `src/` and `src/cli.py` or similar) and identify where `check_shapes` is forwarded; replicate that wiring for `check_devices`, `check_phases`, and `check_gradients`.
  - Add CLI flags `--check-devices`, `--check-phases`, `--check-gradients` (default off, like existing flags) that thread through to the underlying checker registry.
  - Update the README section that currently disclaims these flags to state they are now forwarded, and note the disclaimer is removed.
  - Add a pytest test `tests/test_public_api_check_flags.py` that (a) imports the public API and asserts each flag is accepted and reaches the checker, and (b) invokes the CLI via `subprocess` on a tiny example with each flag, asserting exit code 0 and that the corresponding analysis ran (e.g., a stdout marker or a JSON field).
success_criterion: `python -m pytest tests/test_public_api_check_flags.py -x -q` exits 0 AND `grep -L "currently not forwarded" README.md` succeeds (i.e. the disclaimer string is gone).
fallback_message: If the public API surface cannot be cleanly extended within 10 minutes, the subagent should print `INFEASIBLE: public-API wiring requires deeper refactor` and exit non-zero so the harness reverts.

### Candidate 2: Mechanize the broadcast_add operator lemma in Lean to retire one operator-agnostic axiom
goal: Replace the operator-agnostic-witness axiom for `broadcast_add` with a fully Lean-checked per-operator lemma `applyOpExt_sound_broadcast_add`, so the mechanised composition theorem covers 16 operators with per-operator lemmas (instead of 15) and only `matmul` remains under the agnostic axiom. This raises Soundness by ~0.5 by directly addressing the streak-1 reviewer concern that matmul/broadcast_add are discharged by a 1000-sample property test rather than a Lean proof.
plan:
  - Open `lean/` and locate the existing per-operator lemma pattern (e.g. `applyOpExt_sound_<op>`) and the agnostic-witness statement covering `broadcast_add`.
  - Write `applyOpExt_sound_broadcast_add` as a Lean theorem stating that the rule-table shape function for NumPy/PyTorch broadcast equals the runtime broadcast shape on inputs satisfying the precondition (broadcastability of the trailing-aligned shapes); prove it by case analysis / induction on the shape lists, reusing existing broadcast helpers if any.
  - Update the composition theorem statement to discharge `broadcast_add` via the new lemma rather than the agnostic axiom; keep matmul as the sole agnostic case and update the inline comment/abstract counters in the Lean file accordingly.
  - Ensure `lake build` succeeds with no `sorry` and no new `axiom` introduced for broadcast_add.
success_criterion: `cd lean && lake build` exits 0 AND `grep -R "applyOpExt_sound_broadcast_add" lean/ | grep -v sorry` returns at least one hit AND `grep -R "axiom .*broadcast_add" lean/` returns no matches.
fallback_message: If the broadcast lemma cannot be closed in Lean within 10 minutes, the subagent should print `INFEASIBLE: broadcast_add Lean proof exceeds budget` and exit non-zero so the harness reverts.
```


Changes   +0 -0
Requests  7.5 Premium (40s)
Tokens    ↑ 64.5k • ↓ 2.1k • 48.1k (cached)
