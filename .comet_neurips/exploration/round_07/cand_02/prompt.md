# Role: bold-extension subagent (single attempt)

You are a Sonnet-4.6 subagent attempting ONE speculative extension to
this repo. The harness has snapshotted the working tree before your
run. If you fail, the harness silently reverts and your work
disappears. If you succeed, your changes are merged in and the
extension becomes part of the paper. So: be ambitious, but be HONEST
about the success criterion at the end.

## The candidate to attempt

### Mechanically-extracted HuggingFace bug corpus replacing the 9 hand-distilled cases
goal: Replace the disputed "9/9 naturally-occurring HuggingFace bugs"
       claim — which the reviewer flagged as hand-distilled with no binding
       inclusion rule — with a mechanically-extracted corpus produced by a
       deterministic PR-mining script. Ship a fresh `experiments_v5/hf_natural_bugs_mechanical.json`
       (≥15 bug-PR pairs across ≥5 model families) plus per-bug TG verdicts.
       This lifts Soundness (+0.5) and Contribution (+0.5) by replacing a
       9/9 curated number with an N≥15 mechanically-selected denominator and
       a fully reproducible selection protocol.
plan:
  - Write `scripts/mine_hf_shape_bugs.py` that, given a fixed list of
    HuggingFace transformers commit SHAs (committed in
    `experiments_v5/hf_pr_seed_list.txt`), uses `git log -p` filters
    (regex: `view\(|reshape\(|permute\(|transpose\(|matmul\(` in the diff
    AND keywords `shape|broadcast|dim|size mismatch` in the PR title/body)
    to deterministically select all matching PRs — no human selection.
  - For each selected PR, extract the buggy pre-fix class via AST surgery
    into `experiments_v5/hf_natural_bugs_mechanical/<pr>.py`; record the
    inclusion provenance (commit SHA, regex hit substring) in JSON.
  - Run TensorGuard's existing checker over each extracted class, record
    {pr, family, tg_verdict, ground_truth=BUGGY} into
    `experiments_v5/hf_natural_bugs_mechanical.json`.
  - Add `tests/test_hf_mechanical_corpus.py` asserting (i) ≥15 entries,
    (ii) ≥5 distinct `family` values, (iii) every entry has a non-null
    `tg_verdict` and `provenance.regex_hit`, (iv) the script is
    deterministic (rerun produces byte-identical JSON modulo timestamp).
  - Print a summary line `MECHANICAL_HF: <pass>/<total> across <k> families`.
success_criterion: `python scripts/mine_hf_shape_bugs.py --offline-fixture experiments_v5/hf_pr_seed_list.txt && pytest tests/test_hf_mechanical_corpus.py -x` exits 0 AND `python -c "import json; d=json.load(open('experiments_v5/hf_natural_bugs_mechanical.json')); fams={e['family'] for e in d['entries']}; assert len(d['entries'])>=15 and len(fams)>=5"` exits 0.
fallback_message: If live HF git access is unavailable and no offline fixture of PR diffs can be assembled in 10 minutes, emit `INFEASIBLE: HF PR mining requires network or a pre-staged diff fixture not present in repo` so the harness reverts.
```


Changes   +0 -0
Requests  7.5 Premium (44s)
Tokens    ↑ 74.0k • ↓ 2.5k • 52.4k (cached)

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

Round: 7, candidate 2/2
