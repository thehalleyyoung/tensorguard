● Read brainstorm.prompt.md
  │ .comet_neurips/exploration/round_07/brainstorm.prompt.md
  └ 111 lines read

```
### Candidate 1: Per-block audit table pinning each of the 5 unconditional-RP catches to a specific Lean rule
goal: Ship the missing `experiments_v5/audited_footprint_unconditional_rp.json`
       as a populated artifact (one row per catch) plus a regenerator script
       that derives the table directly from the 488-block run and the Lean
       handler→rule classification. The reviewer has stated explicitly that
       this single deliverable would lift Overall from 5→6; it primarily
       lifts Contribution (+1) and secondarily Soundness (+0.5) by turning a
       summary statistic into a mechanically-checkable subset.
plan:
  - Locate the 488-block real-source corpus run output and the audited handler
    list (49 handlers = 36 Lean + 13 pen-and-paper); identify which 5 blocks
    are simultaneously (i) unconditional-RP=False under empty-assume_M and
    (ii) every handler appearing in the verdict's proof trace is in the
    36-handler Lean-audited set.
  - Write `scripts/build_audited_footprint_table.py` that reads the corpus
    JSON + the handler→Lean-rule map, emits one row per catch with fields:
    `block_id`, `module_path`, `handler_chain`, `lean_rule`,
    `non_audited_handlers` (must be empty list), `verdict`.
  - Have the script write `experiments_v5/audited_footprint_unconditional_rp.json`
    and assert at exit that exactly 5 rows are emitted, each with
    `non_audited_handlers == []` and a non-null `lean_rule`.
  - Add `tests/test_audited_footprint_table.py` that re-runs the script and
    validates the JSON schema + the 5-row + non-audited-empty invariants.
  - Reconcile the 13-vs-15 pen-and-paper count by emitting a sibling file
    `experiments_v5/handler_classification.json` enumerating every handler
    with `lean_audited: bool` so the count is derivable, not asserted.
success_criterion: `python scripts/build_audited_footprint_table.py && pytest tests/test_audited_footprint_table.py -x` exits 0 AND `python -c "import json; d=json.load(open('experiments_v5/audited_footprint_unconditional_rp.json')); assert len(d['catches'])==5 and all(r['non_audited_handlers']==[] and r['lean_rule'] for r in d['catches'])"` exits 0.
fallback_message: If the verdict-trace data needed to attribute each catch to a Lean rule cannot be reconstructed in 10 minutes, emit `INFEASIBLE: cannot derive per-catch handler chain from existing 488-block corpus output without rerunning verifier instrumentation` so the harness reverts.

### Candidate 2: Mechanically-extracted HuggingFace bug corpus replacing the 9 hand-distilled cases
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
