# Role: bold-extension subagent (single attempt)

You are a Sonnet-4.6 subagent attempting ONE speculative extension to
this repo. The harness has snapshotted the working tree before your
run. If you fail, the harness silently reverts and your work
disappears. If you succeed, your changes are merged in and the
extension becomes part of the paper. So: be ambitious, but be HONEST
about the success criterion at the end.

## The candidate to attempt

### torch.compile head-to-head benchmark on the unfiltered post-freeze N=15 sample
goal: Build a reproducible head-to-head benchmark harness that runs TensorGuard, `torch.compile` (FakeTensorMode + dynamo export), and Pytea on the pre-registered post-freeze N=15 unfiltered corpus, emitting a per-bug verdict-pair table plus the BH-adjusted Fisher p-values. Directly answers the "abandoned Pytea baseline" weakness and the "per-bug verdict pair against torch.compile" reviewer question; lifts Contribution by ~1 because it converts a buried paragraph into a load-bearing experiment.
plan:
- Locate the pre-registered post-freeze N=15 sample (search `benchmarks/`, `real_benchmarks/`, `experiments_v5/` for `post_freeze`, `n15`, `unfiltered`); if absent, materialise it from the corpus mining script using the documented seed/freeze date.
- Write `benchmarks/torch_compile_headtohead.py` that, for each of the 15 bug snippets, runs (a) `tensorguard` in unconditional mode, (b) `torch.compile(mode="reduce-overhead")` + `FakeTensorMode` shape inference wrapped in try/except to capture pre-runtime errors, and (c) the existing Pytea harness; record `{bug_id, tg_verdict, compile_verdict, pytea_verdict, ground_truth}` to `benchmarks/results/headtohead_n15.csv`.
- Compute pairwise McNemar exact p-values and BH-adjusted Fisher p-values across the three tools; write to `benchmarks/results/headtohead_n15_stats.json`.
- Add `tests/test_headtohead_n15.py` asserting CSV has exactly 15 rows, all three verdict columns are populated for every row, and the stats JSON parses with the three required keys.
success_criterion: `python -m benchmarks.torch_compile_headtohead --out benchmarks/results/headtohead_n15.csv && pytest tests/test_headtohead_n15.py -q` exits 0 AND `python -c "import csv; rows=list(csv.DictReader(open('benchmarks/results/headtohead_n15.csv'))); assert len(rows)==15 and all(r['tg_verdict'] and r['compile_verdict'] and r['pytea_verdict'] for r in rows)"` exits 0.
fallback_message: If the post-freeze N=15 sample cannot be reconstructed or `torch.compile` cannot be imported in this environment within the budget, emit `INFEASIBLE: headtohead-n15 blocked by <one-line reason>` to stdout and revert all new files via `git clean -fd benchmarks/ tests/ && git checkout -- benchmarks/ tests/`.


Changes   +0 -0
Requests  7.5 Premium (40s)
Tokens    ↑ 67.3k • ↓ 2.4k • 48.2k (cached)

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

Round: 1, candidate 2/2
