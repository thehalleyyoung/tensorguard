# Role: bold-extension subagent (single attempt)

You are a Sonnet-4.6 subagent attempting ONE speculative extension to
this repo. The harness has snapshotted the working tree before your
run. If you fail, the harness silently reverts and your work
disappears. If you succeed, your changes are merged in and the
extension becomes part of the paper. So: be ambitious, but be HONEST
about the success criterion at the end.

## The candidate to attempt

### Extend Lean operator-soundness mechanisation from 17 to ≥25 operators, covering majority CV traffic
goal: Grow the Lean 4 mechanised operator fragment from the current 17 operators to at least 25 by adding `applyOp_sound_*` lemmas (sorry-free) for the 8 highest-CV-traffic handlers currently outside the fragment, then export an updated operator registry JSON consumed by the Python analyser and recompute the "fraction of CV verdicts entirely under Lean-witnessed handlers" headline number. Directly substantiates C2 and answers the round-4 question about the 79-vs-17 gap; expected to lift Contribution +1 and Soundness +1.
plan:
  - From `src/model_checker.py` handler table and any CV-verdict log under `experiments_v5/` or `real_benchmarks/`, rank the 79 handlers by CV-verdict count on the 488-block corpus and pick the top 8 not already in the Lean fragment.
  - For each of those 8 ops, add a `applyOp_sound_<op>` theorem in `lean/` mirroring the existing 17 lemmas' shape (input refinement → output refinement under the same DSL); reuse existing tactics, no new axioms, no `sorry`.
  - Regenerate the operator-registry JSON (whatever script currently exports it) and add a Python check that every handler tagged `lean_audited=True` has a corresponding theorem name in the JSON.
  - Add `reproducibility/cv_lean_coverage.py` that recomputes and prints `<n_cv_in_fragment>/128` after the registry update, and writes the number to `reproducibility/cv_lean_coverage.txt`.
success_criterion: `cd lean && lake build` exits 0 AND `grep -c "^theorem applyOp_sound_" lean/**/*.lean` reports >=25 AND `grep -c "sorry" $(grep -rl "applyOp_sound_" lean/)` reports 0 AND `python reproducibility/cv_lean_coverage.py` writes a file `reproducibility/cv_lean_coverage.txt` containing an integer strictly greater than the pre-existing baseline (record baseline in the same script before the update).
fallback_message: If `lake build` cannot be reached in the budget or the existing Lean DSL lacks primitives for the chosen ops, emit `LEAN_EXTENSION_INFEASIBLE: <one-line root cause>` to stdout, revert any partial Lean edits via `git checkout -- lean/`, and exit non-zero so the harness reverts cleanly.


Changes   +0 -0
Requests  7.5 Premium (1m 2s)
Tokens    ↑ 142.6k • ↓ 3.2k • 119.5k (cached)

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

Round: 4, candidate 2/2
