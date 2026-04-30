# Role: bold-extension subagent (single attempt)

You are a Sonnet-4.6 subagent attempting ONE speculative extension to
this repo. The harness has snapshotted the working tree before your
run. If you fail, the harness silently reverts and your work
disappears. If you succeed, your changes are merged in and the
extension becomes part of the paper. So: be ambitious, but be HONEST
about the success criterion at the end.

## The candidate to attempt

### Theorem-footprint-restricted real-source rerun with new headline table
goal: Add a new evaluation mode that re-runs the 488-block real-source corpus restricted to verdicts whose entire derivation path lies strictly inside the Lean-or-pen-and-paper audited operator footprint, and emit a CSV+JSON artifact with per-block footprint classification (audited / tested-only / out-of-scope) plus aggregate {V, CV, RP, A} counts. This directly addresses the reviewer's borderline condition for a 6 by producing a strictly theorem-backed real-source headline number, lifting Soundness from 2→3 and Contribution from 3→4.
plan:
  - Locate the existing 488-block real-source eval driver under `experiments_v5/` or `real_benchmarks/` and the audited-operator list (Lean fragment + pen-and-paper handlers) referenced by `tab:soundness-footprint-185`.
  - Add a `--footprint-strict` flag to the eval driver that classifies every operator touched on each derivation path and assigns each block to {audited, tested-only-touched, oos-touched}; emit `experiments_v5/footprint_strict_488.csv` with columns `block_id, verdict, footprint_class, ops_touched`.
  - Add a small aggregator script that reads the CSV and writes `experiments_v5/footprint_strict_488_summary.json` with `{audited: {V, CV, RP, A}, total_audited, total_blocks}`.
  - Add `tests/test_footprint_strict_488.py` that runs the driver, asserts the CSV has 488 rows, the JSON sums to 488, and `audited.V + audited.CV + audited.RP > 0`.
success_criterion: `pytest tests/test_footprint_strict_488.py -x` exits 0 AND `experiments_v5/footprint_strict_488.csv` exists with ≥488 data rows AND `experiments_v5/footprint_strict_488_summary.json` parses and contains an `audited` key whose `V+CV+RP+A` sums to the audited-class row count.
fallback_message: If the audited-operator list cannot be located or the eval driver cannot be invoked headlessly within the budget, the subagent should print `INFEASIBLE: footprint-strict rerun blocked` and make no commits so the harness reverts cleanly.

## Time + scope budget

  * Wall-clock: aim for under 10 minutes; use `timeout 60 <cmd>` /
    `timeout 120 <cmd>` etc to bound any single command.
  * Filesystem: edit anywhere in the repo EXCEPT `.comet_neurips/`
    (that's the harness state) and `spawn_sonnet_subagent.sh`.
  * Do NOT modify `neurips.tex` / `main.tex` / `paper.tex` in this
    pass — paper integration happens in the next improver round, not
    here. Your job is to ship code/data/proofs that the next improver
    can fold into the paper.

## What you must do

1. Read the goal, plan, and success_criterion from the candidate
   block above.
2. Implement the plan. Write code, add files, edit modules, etc. Do
   what a real researcher would do.
3. Run the success_criterion command. Capture its exit code.
4. Decide outcome:
   * If the success_criterion command succeeded AND the artifact it
     verified actually exists in the working tree, the outcome is
     **WIN**.
   * Otherwise, the outcome is **FAIL**. (If you got blocked early,
     emit FAIL with the fallback_message.)

## Output format (the harness greps your stdout for these markers)

The LAST non-empty line of your stdout MUST be EXACTLY one of:

  * `EXPLORE_OUTCOME: WIN — <one-sentence description of the win,
    naming concepts not files>`
  * `EXPLORE_OUTCOME: FAIL — <one-sentence reason>`

Before that line, summarize what you did in a `## Attempt log`
section: list the files you created or modified, the commands you
ran with their exit codes, and the success-criterion output. Be
honest. If you fabricated a number rather than measuring it, mark
the outcome as FAIL.

If you cannot determine an honest outcome (e.g. the criterion was
ambiguous), default to FAIL.

Round: 5, candidate 1/2
