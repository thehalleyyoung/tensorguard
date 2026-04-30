● Read round20_reviewer.md
  │ .comet_neurips/round20_reviewer.md
  └ 120 lines read

I’m pulling the paper’s core claims and the repo evidence side by side so the review stays grounded in what’s actually substantiated.

● Read pyproject.toml
  └ L1:220 (44 lines read)

● Read README.md
  └ L1:240 (240 lines read)

● Search (grep)
  │ "11/57|25/128|103/185|44 tested-only|28 of 79|488|206|92\.2%|28,000|Theorem
  │ 5|C4|C6|mutation|conv2d|einsum|Dynamo|transformer|handler" (neurips.txt,
  │ neurips.tex, docs)
  └ 8 lines found

● Read neurips.tex
  └ L1:140 (97 lines read)

● Search (grep)
  │ "section\{4\.4|subsection\{4\.4|Handler-scope|Table 7|Theorem
  │ 5|C4|C6|mutation|conv2d|einsum|28,000|488-block|206 refutations|92\.2|1/4
  │ transformer|9 CNN|BH correction|power" (neurips.tex)
  └ No matches found

● Search (glob)
  │ "tests/**/*.py"
  └ 118 files found

● Search (glob)
  │ "experiments_v5/**/*"
  └ 14 files found

I found the abstract’s updated partition counts; now I’m checking whether the evaluation artifacts and tests actually support the paper’s harder claims, especially the held-out test, mutation rate, Dynamo coverage, and boundary sampling.

● Search (glob)
  │ "lean/**/*"
  └ 94 files found

● Search (grep)
  │ "488|185|128|57|Contract-Violation|Verified|Refuted-Proof|tested-only|Lean|p
  │ en-and-paper|soundness scope|TCB" (neurips.txt)
  └ 9 lines found

● Search (glob)
  │ "src/**/*.py"
  └ 248 files found

● Search (grep)
  │ "32/57|30/128|62/185|66/185|57/185|206|488|92\.2|118/128|28,000|BH|Benjamini
  │ |McNemar|conv2d|einsum|1/4 transformer|9 CNN|boundary|off-envelope|uniformly
  │ within|power calculation|26|77" (experiments_v5, tests, src, docs)
  └ 8 lines found

● Search (grep)
  │ "32/57|30/128|62/185|66/185|57/185"
  └ 8 lines found

● Search (grep)
  │ "118/128|92\.2%|92\.2"
  └ 7 lines found

● Search (grep)
  │ "26–77|26-77|additional samples|BH correction|Benjamini|power
  │ calculation|adjusted p=1\.00|p=1\.00|N=15|unfiltered post-freeze"
  └ 7 lines found

● Search (grep)
  │ "28,000|28000|off-envelope|boundary check|boundary coverage|10 of
  │ 28|10/28|uniformly within the in-fragment envelope"
  └ 7 lines found

● Read .pdftxt.tmp
  └ L110:170 (61 lines read)

● Read .pdftxt.tmp
  └ L820:900 (80 lines read)

● Read .pdftxt.tmp
  └ L900:980 (81 lines read)

● Read .pdftxt.tmp
  └ L690:760 (71 lines read)

● Search (grep)
  │ "boundary|off-envelope|in-envelope|10 of 28|10/28|28,000|28000|rule
  │ table|sample" (.pdftxt.tmp, neurips.txt, docs/paper)
  └ 7 lines found

● Search (grep)
  │ "Theorem 5|C4|Dynamo|guard|surrogate|transformer|CNN block|9/|1/4|4
  │ transformer|correspondence" (.pdftxt.tmp, neurips.txt, docs/paper)
  └ 8 lines found

● Search (grep)
  │ "mutation|kill rate|7/50|14%|conv2d|einsum|0/10|load-bearing|extended
  │ suite|18-case|targeted" (.pdftxt.tmp, neurips.txt, docs/paper,
  │ experiments_v5)
  └ 6 lines found

The extracted PDF text is noisy, so I’m switching to narrower probes around the exact evaluation claims instead of broad grep dumps.

● Search (grep)
  │ "9 CNN|1/4 transformer|4 transformer|surrogate|Dynamo-guard|Theorem 5"
  │ (.pdftxt.tmp)
  └ 129 lines found

● Search (grep)
  │ "28,000/28,000|10 of 28|10/28|off-envelope" (.pdftxt.tmp)
  └ 31 lines found

● Search (grep)
  │ "7/50|14%|conv2d 0/10|einsum 0/10|18-case targeted|load-bearing
  │ handlers|mutation" (.pdftxt.tmp)
  └ 51 lines found

● Search (grep)
  │ "BH correction|adjusted p=1.00|additional samples|26|77|post-freeze|N
  │ =6|N=6|generalisation" (.pdftxt.tmp)
  └ 8 lines found

● Read .pdftxt.tmp
  └ L1:120 (120 lines read)

● Read .pdftxt.tmp
  └ L1865:1915 (51 lines read)

● Read .pdftxt.tmp
  └ L1506:1540 (35 lines read)

● Read .pdftxt.tmp
  └ L1640:1660 (21 lines read)

● Search (grep)
  │ "3/6 Refuted-Proof|post-freeze headline is 3/6|N =6 post-freeze|five-way
  │ verdict taxonomy|five-way" (.pdftxt.tmp)
  └ 22 lines found

● Search (grep)
  │ "Table 7|11 of the 57|25 of the
  │ 128|32/57|30/128|62/185|36/185|105/185|105/185|57/185|tested-only"
  │ (.pdftxt.tmp, docs/paper)
  └ 7 lines found

● Search (grep)
  │ "5/15|15|FakeTensorMode 2/15|Pytea 3/15|post-freeze real-PR
  │ sample|directional result|statistically separable|alpha=0.05" (.pdftxt.tmp,
  │ docs/paper)
  └ 8 lines found

● Search (grep)
  │ "conv2d|einsum|load-bearing|targeted extension|18-case|ext_|mutation-testing
  │ rate" (.pdftxt.tmp, experiments_v5, docs/paper)
  └ 8 lines found

I’ve confirmed one serious paper issue already: the abstract’s revised handler-scope partition no longer matches the body’s Section 4.4/Table 7 partition. I’m checking whether that inconsistency is isolated or tied to the released reproducibility artifacts.

● Search (grep)
  │ "11 of the 57|25 of the 128|33 \+ 72=105/185|32/57|30/128|62/185|57/185|48
  │ tested-only|35 remaining" (docs/paper/sections_v5, neurips.tex)
  └ 39 lines found

● Search (grep)
  │ "joint-realisability|118/128|92\.2|assume_M|assume M"
  │ (docs/paper/sections_v5, neurips.tex)
  └ 22 lines found

● Search (glob)
  │ "docs/paper/sections_v5/*.tex"
  └ 29 files found

● Search (glob)
  │ "reproducibility/**/*scope*"
  └ 12 files found

● Search (glob)
  │ "reproducibility/**/*handler*"
  └ 17 files found

The reproducibility folder looks useful here: it contains direct handler-scope and post-freeze artifacts, so I’m checking whether the released numbers match the paper’s current prose.

● Read handler_scope_per_block.md
  │ reproducibility/handler_scope_per_block.md
  └ L1:220 (73 lines read)

● Read post_freeze_in_soundness_scope.md
  │ reproducibility/post_freeze_in_soundness_scope.md
  └ L1:220 (38 lines read)

● Read surviving_mutants_handler_classification.md
  │ reproducibility/surviving_mutants_handler_classification.md
  └ L1:220 (89 lines read)

● Read postfreeze_5catches_handler_scope.md
  │ reproducibility/postfreeze_5catches_handler_scope.md
  └ L1:220 (20 lines read)

● Read handler_soundness_table.tex
  │ docs/paper/sections_v5/handler_soundness_table.tex
  └ L1:220 (96 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1368:1445 (78 lines read)

● Read E_dynamo.tex
  │ docs/paper/sections_v5/E_dynamo.tex
  └ L1:220 (109 lines read)

● Read F_benchmark.tex
  │ docs/paper/sections_v5/F_benchmark.tex
  └ L1:260 (148 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:180 (180 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L240:360 (121 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1180:1320 (141 lines read)

● Search (grep)
  │ "Post-freeze generalisation|post-freeze|3/6|5/15|Theorem 5|14 importable
  │ nn.Module subclasses|9 CNN|4 transformer|28,000/28,000|Mutation-testing
  │ rate" (docs/paper/sections_v5/eval_v6.tex,
  │ docs/paper/sections_v5/intro_v6.tex, docs/paper/sections_v5/limconc_v6.tex)
  └ 7 lines found

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1020:1205 (185 lines read)

● Search (grep)
  │ "28,000/28,000|10 rules|2,400 off-envelope|boundary check|sampled uniformly
  │ within" (docs/paper/sections_v5/eval_v6.tex)
  └ 15 lines found

● Search (grep)
  │ "14 importable|9 CNN|4
  │ transformer|surrogate|Theorem~\\ref\{thm:dynamo-corr\}|falsifier|0
  │ SHAPE/DTYPE/RANK|72 INT recompiles|55-module|67-module"
  │ (docs/paper/sections_v5/eval_v6.tex)
  └ 71 lines found

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ L90:120 (31 lines read)

I have enough paper text to score the claims, but I’m running a few targeted repo tests/artifact checks for the most consequential evaluation sections so I can distinguish “stated in the paper” from “actually backed by code.”

● Search (grep)
  │ "post[-_ ]freeze|3/6|7/10|53/60|32/34|118/128|Theorem
  │ 5|28,000|mutation|7/50" (tests)
  └ 9 lines found

● Run targeted evaluation tests (shell)
  │ python -m pytest -q tests/test_statistical_rigor.py
  │ tests/test_dynamo_gap_analysis.py tests/test_lean_handler_parity.py
  └ 2 lines...

● Run key reproducibility scripts (shell)
  │ python reproducibility/handler_scope_per_block.py && python
  │ reproducibility/surviving_mutants_handler_classification.py >/dev/null &&
  │ python reproducibility/post_freeze_in_soundness_scope.py >/dev/null
  └ 2 lines...

The repo is set up with `python3`, not `python`; I’m rerunning the checks with the right interpreter.

● Search (grep)
  │ "def test_|class Test|118/128|7/50|3/6|53/60|32/34|28,000|Theorem
  │ 5|post_freeze" (tests/test_statistical_rigor.py,
  │ tests/test_dynamo_gap_analysis.py, tests/test_lean_handler_parity.py)
  └ 9 lines found

● Run targeted evaluation tests (shell)
  │ python3 -m pytest -q tests/test_statistical_rigor.py
  │ tests/test_dynamo_gap_analysis.py tests/test_lean_handler_parity.py
  └ 13 lines...

● Run key reproducibility scripts (shell)
  │ python3 reproducibility/handler_scope_per_block.py && python3
  │ reproducibility/surviving_mutants_handler_classification.py >/dev/null &&
  │ python3 reproducibility/post_freeze_in_soundness_scope.py >/dev/null
  └ 6 lines...

The targeted tests and scripts back the existence of the evaluation machinery, but they also surfaced a likely artifact-management problem: some reproducibility files were regenerated and need a quick status check so I don’t misread stale vs current numbers.

● Read handler_scope_per_block.py
  │ reproducibility/handler_scope_per_block.py
  └ L1:220 (220 lines read)

● Inspect git status and diffs (shell)
  │ git --no-pager status --short && printf '\n---\n' && git --no-pager diff --
  │ reproducibility/handler_scope_per_block.md
  │ reproducibility/handler_scope_per_block.json
  │ reproducibility/surviving_mutants_handler_classification.json
  │ reproducibility/post_freeze_in_soundness_scope.json | sed -n '1,220p'
  └ 279 lines...

## Summary
This paper presents TensorGuard, a static refinement-type checker for PyTorch `nn.Module` code that aims to verify shape consistency and a restricted notion of gradient-flow correctness without executing the model. The technical core is a refinement calculus with assume/guarantee contracts at module boundaries, plus a verdict taxonomy that separates unconditional refutations from contract-conditional findings and conservative warnings. The evaluation emphasizes three axes: a 60-bug historical shape-bug corpus, a 488-block real-library corpus, and a Lean-audited operator-rule table with empirical parity checks against PyTorch. The paper also argues for a one-directional correspondence between TensorGuard’s refinement variables and TorchDynamo guard bits, and it is commendably explicit that this is a necessary-direction statement rather than an equivalence. Overall, the work is ambitious and substantially more careful than many systems papers about stating what is and is not covered by the formal claims.

## Prior weakness disposition
- [PARTIAL] The handler-scope arithmetic in §4.4 and the abstract still does not reconcile. Abstract says "11/57 Verified and 25/128... -- The source and current reproducibility notes now support a reconciled 62/185 partition, but the provided PDF artifact still contains the older contradictory counts.
- [PARTIAL] The only pre-registered unbiased generalisation test (N=15 unfiltered post-freeze) remains statistically null after BH correction... -- The current source de-emphasizes that null result and replaces it with a smaller frozen post-freeze analysis, but the generalisation evidence is still not decisive.
- [PARTIAL] The headline analyser-wide mutation-kill rate is 7/50 (14%) on the union of three corpora. The two zero-kill... -- The targeted conv2d/einsum extension addresses the zero-kill handler complaint, but the analyzer-wide headline remains 7/50 and the strongest rescue evidence comes from a tailored add-on corpus.
- [PARTIAL] C4 (Dynamo-guard inclusion, Theorem 5) is empirically instantiated end-to-end without surrogate on 9 CNN blocks but only 1/4 transformer blocks... -- The paper now scopes C4 more carefully and adds more audit detail, but end-to-end transformer evidence is still limited because the transformer blocks remain surrogate-based.
- [PARTIAL] The 488-block "headline" is "0 unconditional RP," with all 206 refutations being CV (synthesised caller-rely) or LW... -- This limitation is now stated much more clearly and used to calibrate the claim, but the empirical limitation itself is unchanged.
- [UNRESOLVED] C6's "28,000/28,000 agree with torch 2.9.1" is sampled "uniformly within the in-fragment envelope" of each rule... -- The paper still reports off-envelope checks for only 10 of the 28 audited rules, so the boundary-validity concern remains.
- [PARTIAL] Several of the most consequential numbers in the abstract — 11/57, 25/128, 103/185, 44 tested-only, 28 of 79 handlers... -- The source now centralizes a cleaner 62/66/57 partition, but the shipped artifacts still disagree across PDF, source, and reproducibility files.

## Strengths
- The paper is unusually disciplined about calibration: it explicitly distinguishes RP, CV, LW, Abstain, and N/A, and it openly states that the 488-block corpus yields **0 unconditional RP**.
- The 60-bug historical corpus result is strong, and the fragment-fair comparison against Pytea is much better motivated than a raw apples-to-oranges benchmark.
- The formalization effort is substantial: a refinement calculus, module-level assume/guarantee reasoning, and a nontrivial Lean audit of 28 operator rules with 11 previously axiomatic lemmas closed sorry-free.
- The repository contains real reproducibility structure rather than only headline tables; the evaluation is tied to scripts, handler-scope audits, and parity/property tests.

## Weaknesses
- The release is internally inconsistent on core evaluation numbers. In particular, `neurips.tex` / `docs/paper/sections_v5/eval_v6.tex` present the reconciled `32/57`, `30/128`, `62/185`, `66/185`, `57/185` partition, while the shipped PDF text still contains the older `11/57`, `25/128`, `36/185` bookkeeping, and related reproducibility notes still mention older post-freeze counts.
- On natural library code, the headline remains **0/488 unconditional RP**; all 206 block-corpus refutations are CV or LW. That means the strongest live evidence of actual bug-finding still comes from curated bug corpora, not from the main 488-block real-source benchmark.
- The Theorem 5 evidence is still weak on the transformer side. The paper’s own current source treats the **10 fully end-to-end CNN-type subjects** as the headline falsifier test, while the four transformer blocks are audited via documented forward-signature surrogates rather than full end-to-end instantiation.
- Section 4.4’s `28,000/28,000` agreement is mainly an implementation-parity check because sampling is inside the declared envelopes; the harder question is whether those envelopes are right, and the off-envelope check still covers only **10/28** rules.
- The analyzer-wide mutation result is still modest at **7/50** on the union of the three main corpora. The more reassuring conv2d/einsum numbers come from a targeted extension corpus, which is useful but materially less convincing than seeing the headline mutation rate move.
- The post-freeze generalisation story remains hard to trust as presented: the current source text reports a small `3/6` upstream-faithful post-freeze result, while stale repo artifacts still advertise older `5/15` / `6-fire` framings.

## Questions
- Which artifact should reviewers treat as authoritative for the handler-scope and post-freeze numbers: the current source, the shipped PDF, or the reproducibility markdown/json outputs? Please provide one canonical table and explain why the others disagree.
- Can the authors provide an end-to-end, non-surrogate transformer-block audit for Theorem 5, or else narrow the framing of C4 so it is explicitly a CNN-dominant result?
- Why is the boundary check in Section 4.4 still limited to 10 of the 28 audited rules, and what prevented running the same off-envelope procedure for the full audited set?
- For the mutation study, how many of the 18 “structurally false-RP-capable” surviving mutants are actually exercised on the paths responsible for the paper’s key RP/CV results?

## Scores
Soundness: 2
Presentation: 2
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
A single clean regeneration of the evaluation artifact set would raise my score by one point: rebuild the PDF from the current source and make the PDF, source, and reproducibility outputs agree on the handler-scope, post-freeze, and theorem-audit numbers. Right now, the largest blocker is not lack of technical ambition but lack of confidence that the released artifacts support one stable story.


Changes   +0 -0
Requests  1 Premium (4m 18s)
Tokens    ↑ 1.0m • ↓ 13.0k • 962.7k (cached) • 5.6k (reasoning)
