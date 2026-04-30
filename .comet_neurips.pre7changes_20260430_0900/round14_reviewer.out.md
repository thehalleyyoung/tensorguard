● Read round14_reviewer.md
  │ .comet_neurips/round14_reviewer.md
  └ 120 lines read

I’m pulling the paper text and the specific evidence behind the previous reviewer’s objections so the disposition labels are tied to what’s actually in the current artifact.

● Read README.md
  └ L1:240 (240 lines read)

● Read API.md
  └ L1:240 (240 lines read)

● Search (grep)
  │ "^(#|##|\\section|\\subsection)|Theorem 2|Theorem 5|Table 5|488|free-symboli
  │ c|cross-family|mutation|decoder|Verified|Abstain|RP|gradient-flow|CEGAR"
  │ (neurips.txt, neurips.tex, docs)
  └ 8 lines found

● Read neurips.tex
  └ L1:220 (89 lines read)

● Search (grep)
  │ "\\(section|subsection|input)\{|Theorem 2|Theorem 5|Table 5|table\{|label\{t
  │ ab:|label\{thm:|488|free-symbolic|cross-family|decoder|mutation|stress
  │ bench|high-confidence|Contract-Violation|user-visible" (neurips.tex)
  └ 12 lines found

The abstract already answers part of the prior reviewer’s complaint: it now explicitly narrows what the 488-block corpus supports and states that the free-symbolic regime yields zero unconditional refuted proofs. I’m checking whether the later sections and scripts genuinely back those numbers.

✗ Search (grep)
  │ "Theorem 2|Theorem 5|Table
  │ 5|488|free-symbolic|cross-family|decoder|mutation|stress
  │ benchmark|Contract-Violation|Refuted-Proof|head-to-head|historical
  │ corpus|fragment-fair|CNN|transformer|26 modules|57
  │ Verified|25/128|53/60|7/50" (sections_v5, docs/paper/sections_v5)
  └ rg: /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
    sections_v5: No such file or directory (os error 2)

● Search (glob)
  │ "**/sections_v5/*.tex"
  └ 29 files found

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:320 (320 lines read)

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ L1:240 (186 lines read)

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ L1:240 (135 lines read)

● Search (grep)
  │ "Theorem 2|Theorem 5|Table
  │ 5|488|free-symbolic|cross-family|decoder|mutation|stress
  │ benchmark|Contract-Violation|Refuted-Proof|head-to-head|historical
  │ corpus|fragment-fair|CNN|transformer|26 modules|57
  │ Verified|25/128|53/60|7/50|43/50|14%" (docs/paper/sections_v5)
  └ 7 lines found

I’ve got the paper-side story. Next I’m checking whether the released artifacts actually expose the same counts and caveats the text now claims, especially for the free-symbolic rerun, the real-bug re-extracts, and the ablation/mutation results.

● Read pyproject.toml
  └ L1:220 (44 lines read)

● Search (grep)
  │ "57|34|23|128|206|488|53/60|7/10|1/4|9/|25-case|ablation|mutation|7/50|43/50
  │ |cross-family|decoder|free-symbolic|symbolic-config|BartLearned|ChannelAtten
  │ tion|FalconLinear|real public-repo bug|post-freeze|hybrid" (experiments_v5,
  │ reproducibility, tests)
  └ 7 lines found

● Search (glob)
  │ "experiments_v5/**/*"
  └ 14 files found

● List directory experiments_v5
  └ 51 files found

● Search (grep)
  │ "free-symbolic|symbolic-config|57 Verified|34 V|23|128 CV|206
  │ LW|53/60|7/10|post-freeze|25-case|hybrid|mutation|7/50|43
  │ survivors|cross-family|decoder|26
  │ modules|BartLearnedPos|ChannelAttention|FalconLinear" (experiments_v5,
  │ reproducibility)
  └ 9 files found

● Read feature_ablation.json
  │ experiments_v5/feature_ablation.json
  └ L1:260 (175 lines read)

● Read track_D_summary.md
  │ experiments_v5/track_D_summary.md
  └ L1:240 (58 lines read)

● Read track_E_summary.md
  │ experiments_v5/track_E_summary.md
  └ L1:240 (59 lines read)

● Read verdict_reclassification.json
  │ experiments_v5/verdict_reclassification.json
  └ L1:260 (260 lines read)

● Read v5_benchmark_results.json
  │ experiments_v5/v5_benchmark_results.json
  └ L1:260 (260 lines read)

● Read hybrid_mode_results.json
  │ experiments_v5/hybrid_mode_results.json
  └ L1:260 (260 lines read)

● Read post_freeze_n15_precision_recall.json
  │ reproducibility/post_freeze_n15_precision_recall.json
  └ L1:220 (220 lines read)

● List directory reproducibility
  └ 270 files found

● Search (grep)
  │ "free-symbolic|free symbolic|34 Verified|23|57
  │ Verified|206|128|78|295|15/488|15/295|0/488|0
  │ unconditional|single-default-omitted|118/128|92.2" (experiments_v5,
  │ reproducibility)
  └ 9 files found

● Read cv_caller_rely_joint_sat_full128.md
  │ reproducibility/cv_caller_rely_joint_sat_full128.md
  └ L1:220 (31 lines read)

● Read no_assume_subset_488.md
  │ reproducibility/no_assume_subset_488.md
  └ L1:220 (32 lines read)

● Read lw_rp_gap.md
  │ reproducibility/lw_rp_gap.md
  └ L1:260 (29 lines read)

● Read mutation_kill_rate_corpora.md
  │ reproducibility/mutation_kill_rate_corpora.md
  └ L1:220 (51 lines read)

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ L40:180 (132 lines read)

● Read real_corpus_ablation.md
  │ reproducibility/real_corpus_ablation.md
  └ L1:220 (31 lines read)

● Search (grep)
  │ "label\{thm:soundness\}|Theorem|Verified|Contract-Violation|Refuted-Proof|Li
  │ brary-Warn|assume_M|reduction" (docs/paper/sections_v5/calculus_v6.tex)
  └ 18 lines found

● Read dynamo_theorem5_n100.md
  │ reproducibility/dynamo_theorem5_n100.md
  └ L1:240 (18 lines read)

● Read hf_extra_families_round11.md
  │ reproducibility/hf_extra_families_round11.md
  └ L1:220 (83 lines read)

● Read E_dynamo.tex
  │ docs/paper/sections_v5/E_dynamo.tex
  └ L1:260 (109 lines read)

● Read dynamo_e2e_15modules.md
  │ reproducibility/dynamo_e2e_15modules.md
  └ L1:220 (52 lines read)

● Search (grep)
  │ "cross-family|decoder|Llama|Qwen2|Mistral|Gemma|Phi-3|26
  │ modules|buggy|Theorem 5|thm:dynamo-corr|surrogate|1/4|nine CNN|four
  │ transformer|end-to-end|Track E" (docs/paper/sections_v5, reproducibility,
  │ experiments_v5)
  └ 8 lines found

● Read hf_extra_model_family.md
  │ reproducibility/hf_extra_model_family.md
  └ L1:220 (60 lines read)

● Read track_C_summary.md
  │ experiments_v5/track_C_summary.md
  └ L1:260 (21 lines read)

● Read hf_extra_family_round_comet1.md
  │ reproducibility/hf_extra_family_round_comet1.md
  └ L1:220 (32 lines read)

I found one potentially important artifact-level issue: the released `experiments_v5/v5_benchmark_results.json` reports **50 Verified / 213 Refuted / 225 Abstain** on the 488-block corpus, while the paper’s headline tables and several other artifacts use **57 / 206 / 225**. I’m checking whether the repo explicitly reconciles that mismatch before I lean on it in the review.

● Read paper_artifact_reconciliation.md
  │ reproducibility/paper_artifact_reconciliation.md
  └ L1:260 (60 lines read)

● Read paper_artifact_reconciliation.py
  │ reproducibility/paper_artifact_reconciliation.py
  └ L1:260 (245 lines read)

● Search (grep)
  │ "50|213|57|206|v5_benchmark_results|reconciliation|artifact"
  │ (reproducibility, docs/paper/sections_v5, experiments_v5)
  └ 7 lines found

● Read run_verdict_reclassification.py
  │ experiments_v5/run_verdict_reclassification.py
  └ L1:260 (157 lines read)

● Read run_v5_benchmark.py
  │ experiments_v5/run_v5_benchmark.py
  └ L1:260 (260 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L320:640 (321 lines read)

I narrowed the artifact mismatch: it may be a stale summary field rather than a disagreement in the per-item data. I’m doing one quick count so I don’t overstate it.

● Search (grep)
  │ ""bucket": "Refuted"" (experiments_v5/v5_benchmark_results.json)
  └ 1 file found

● Search (grep)
  │ ""bucket": "Verified"" (experiments_v5/v5_benchmark_results.json)
  └ 1 file found

● Search (grep)
  │ ""bucket": "Abstain"" (experiments_v5/v5_benchmark_results.json)
  └ 1 file found

● Read v5_benchmark_results.json
  │ experiments_v5/v5_benchmark_results.json
  └ L1:120 (120 lines read)

● Count per-input vs summary benchmark buckets (shell)
  │ python - <<'PY'
  │ import json
  │ p='experiments_v5/v5_benchmark_results.json'
  │ with open(p) as f:
  │ d=json.load(f)…
  └ 10 lines...

● Count per-input vs summary benchmark buckets (shell)
  │ python3.11 - <<'PY'
  │ import json
  │ from collections import Counter
  │ with open('experiments_v5/v5_benchmark_results.json') as f:
  │ d=json.load(f)…
  └ 12 lines...

● Read mutation_kill_rate_60bug.md
  │ reproducibility/mutation_kill_rate_60bug.md
  └ L1:220 (32 lines read)

● Read contemporary_baseline_34.md
  │ reproducibility/contemporary_baseline_34.md
  └ L1:220 (77 lines read)

● Read surviving_mutants_handler_classification.md
  │ reproducibility/surviving_mutants_handler_classification.md
  └ L1:220 (89 lines read)

One implementation detail is more consequential than it first looked: the ablation artifact explicitly says several advertised knobs are API no-ops in the current path. I’m checking the wrapper code so I can characterize that accurately rather than just quoting the summary.

● Read main.py
  │ src/cli/main.py
  └ L1:260 (260 lines read)

● Read api.py
  │ src/api.py
  └ L1:260 (259 lines read)

● Search (grep)
  │ "def verify_architecture|check_devices|check_phases|check_gradients|max_cega
  │ r_iterations|high_confidence_only" (src)
  └ 64 lines found

● Read api.py
  │ src/api.py
  └ L680:980 (301 lines read)

● Read api.py
  │ src/api.py
  └ L995:1035 (41 lines read)

## Summary
This paper presents TensorGuard, a static verifier for PyTorch `nn.Module` class source that aims to reason about tensor shapes and gradient-flow properties without instantiating the module, tracing execution, or requiring annotations. The technical core is a refinement-typed calculus with assume/guarantee contracts at module boundaries, a five-way verdict taxonomy (`Verified`, `Refuted-Proof`, `Contract-Violation`, `Library-Warn`, `Abstain`), and a partially mechanized Lean audit of the operator-rule table. Empirically, the paper claims strong performance on a curated 60-bug historical corpus, a fragment-fair head-to-head win over Pytea on a 34-bug subset, and calibrated but limited behavior on a 488-block real-source corpus where unconditional refutations disappear in the user-visible free-symbolic regime. It also reports exploratory TorchDynamo correspondence, a backward grad-flag verifier, and cross-family evaluations on several decoder families beyond the main corpus. The paper is much more careful than average about stating what is and is not covered by the soundness theorem and by the audited fragment.

## Prior weakness disposition
- [RESOLVED] The headline pitch in the abstract — "TENSORGUARD is a no-execution refinement-type checker..." sits awkwardly ag... -- The abstract now explicitly says the 488-block corpus yields 0 unconditional RP in the free-symbolic regime and frames that surface as coverage rather than bug-finding.
- [PARTIAL] The cross-family decoder evaluation (Llama/Qwen2/Mistral/Gemma/Phi-3, 26 modules) still finds zero real cross-fami... -- The breadth is improved to 26 modules across 5 families, but the refutations are still five injected buggy variants plus conservative/known-limitation cases rather than naturally occurring cross-family bugs.
- [PARTIAL] The 488-block Verified count rests on a synthesised symbolic-config envelope; under the free-symbolic regime 23 of ... -- The paper now foregrounds the 34/57 survival and 0-RP free-symbolic result, but the main Verified headline still leans on synthesized config envelopes and the theorem text remains stronger than the empirical user-visible regime.
- [PARTIAL] Theorem 5's empirical surface remains thin where it matters most. The CNN-only restriction (10 subjects, 13 SHAPE r... -- The result is now clearly scoped as exploratory and supplemented with larger audits, but transformer end-to-end evidence is still sparse and partly surrogate-based.
- [PARTIAL] The analyser-wide mutation-kill rate is still 7/50 = 14% at the union of three corpora, with 43 survivors of which... -- The paper now characterizes the survivors and presents the union kill rate honestly, but the measured robustness signal is still modest.
- [RESOLVED] Section 4.1's Table 5 reports that on the real 488-block + 60-bug aggregate every analyser knob (CEGAR, device, ... -- The current paper now explicitly says the real-corpus ablation is flat and limits the claimed contribution to the few knobs that move the hand-designed stress benchmark.
- [PARTIAL] The 60-bug corpus filter (≤40-line self-contained CPU repro raising the cited RuntimeError) selects strongly for b... -- The added upstream-faithful 10-bug and unfiltered 15-PR evaluations help, but the 53/60 figure still comes from a strongly curated in-fragment corpus.

## Strengths
- The paper is unusually honest about calibration: the five-way verdict taxonomy, explicit abstain reasons, and repeated caveats about what RP/CV/LW do and do not mean are all strong.
- The class-source/no-execution operating point is genuinely interesting; on the real-source corpus the execution-based baselines are mostly structurally inapplicable, so this is not a trivial restatement of FakeTensor-style checking.
- The formal side is substantial for an ML systems paper: a refinement-typed calculus, assume/guarantee composition, and a partial Lean audit with a clearly stated trusted computing base.
- The evaluation has improved beyond a single curated corpus by adding upstream-faithful real-public bugs, post-freeze cases, and cross-family decoder coverage, even if those additions do not fully close the empirical gaps.

## Weaknesses
- The most important practical limitation remains the natural-distribution result: in the user-visible free-symbolic regime on the 488-block corpus, the paper reports **0 unconditional Refuted-Proof** verdicts, and 23 of the 57 `Verified` rows collapse to `Abstain`; this makes the real-library evidence much weaker than the curated-bug evidence.
- The cross-family decoder study is still not convincing as a bug-finding result: in `docs/paper/sections_v5/eval_v6.tex` and the corresponding `reproducibility/hf_extra_*.md` artifacts, the RP counts are driven by deliberately broken variants plus `LlamaAttention`'s conservative division guard, while `Phi3SdpaAttentionFusedQKV` is itself logged as a known false-positive RP.
- Theorem 5 remains exploratory rather than decisive. The main table in `docs/paper/sections_v5/E_dynamo.tex` uses mostly `trusted` contracts rather than TG-verified ones, and the released larger audits (`reproducibility/dynamo_theorem5_n100.md`) still have substantial timeout/warmup attrition while testing only the absence of out-of-catalogue SHAPE/DTYPE/RANK guards.
- The robustness story is still thin: `reproducibility/mutation_kill_rate_corpora.md` reports only **7/50** mutants killed at the union of three corpora, and `reproducibility/surviving_mutants_handler_classification.md` says **18** surviving mutants still lie on potentially verdict-emitting paths.
- The released artifacts are not fully audit-friendly on the headline benchmark numbers: `experiments_v5/v5_benchmark_results.json` reports **50 Verified / 213 Refuted / 225 Abstain** on the 488-block corpus, while the paper and other artifacts (`feature_ablation.json`, `hybrid_mode_results.json`) use **57 / 206 / 225**. Even if there is a benign explanation, this weakens confidence in the benchmark bookkeeping.
- The comparative contribution is narrower than the headline bug-catch numbers suggest: `reproducibility/contemporary_baseline_34.md` reports `torch.compile`+FakeTensor catches **34/34** bugs on the same modern subset where TG gets **32/34**, so the real advantage is chiefly the no-instantiation/no-input regime, not absolute bug-catching power when executable harnesses exist.

## Questions
- Which released artifact should a reader treat as authoritative for the 488-block headline counts, and how should the discrepancy between `v5_benchmark_results.json` and the 57/206/225 numbers be interpreted?
- Can the authors provide a real cross-family bug benchmark (e.g., naturally occurring upstream bugs or bug-fix commits in these decoder families) rather than relying mainly on injected negative controls?
- How should readers formally connect Theorem 2 / `thm:soundness` to the empirical `Verified` rows that depend on synthesized config envelopes and partly disappear under the free-symbolic regime?
- For the Dynamo correspondence claim, how many transformer-family modules can be audited end-to-end without surrogates if the time budget is increased, and does the conclusion materially change on that slice?

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
A stronger real-world validation would move this up by one point: specifically, a benchmark with several naturally occurring post-freeze or cross-family upstream bugs caught without injected variants, plus a clean reconciliation of the 488-block artifact counts. Right now the theory is interesting and the paper is careful, but the practical bug-finding evidence on natural-distribution code is still too limited for me to cross the accept bar.


Changes   +0 -0
Requests  1 Premium (3m 27s)
Tokens    ↑ 1.3m • ↓ 12.0k • 1.2m (cached) • 5.1k (reasoning)
