● Read round05_reviewer.md
  │ .comet_neurips/round05_reviewer.md
  └ 294 lines read

I’m pulling the paper’s key claims, theorem/proof scaffolding, and the artifact-backed evaluation tables together so the review tracks the current version rather than the prior round.

● Read neurips.tex
  └ L1:220 (103 lines read)

● Read audited_footprint_unconditional_rp.json
  │ reproducibility/audited_footprint_unconditional_rp.json
  └ 423 lines read

● Read v5_benchmark_results.json
  │ experiments_v5/v5_benchmark_results.json
  └ L1:260 (260 lines read)

● Read pytea_fragment_fair.json
  │ reproducibility/pytea_fragment_fair.json
  └ 567 lines read

● Search (grep)
  │ "\\input\{|\\section\{|\\subsection\{|26/356|15/488|32/34|25/34|McNemar|Wils
  │ on|Theorem|Lemma|Proposition|assume/guarantee|contravariant|covariant|Pytea|
  │ headline|bookkeeping" (neurips.tex, docs/paper)
  └ 8 lines found

I found the core artifact files already, but the remaining decision hinges on how the paper states its theorem and how tightly the evaluation prose matches the released JSON. I’m reading those sections directly now.

✗ Read calculus_v6.tex
  │ sections_v5/calculus_v6.tex
  └ Path does not exist

✗ Read eval_v6.tex
  │ sections_v5/eval_v6.tex
  └ Path does not exist

✗ Read appendix_v6.tex
  │ sections_v5/appendix_v6.tex
  └ Path does not exist

✗ Read limconc_v6.tex
  │ sections_v5/limconc_v6.tex
  └ Path does not exist

● Search (glob)
  │ "**/sections_v5/{intro_v6,calculus_v6,impl_v6,eval_v6,limconc_v6,appendix_v6
  │ }*"
  └ 6 files found

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ L1:260 (142 lines read)

● Read impl_v6.tex
  │ docs/paper/sections_v5/impl_v6.tex
  └ L1:320 (86 lines read)

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ L1:360 (360 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:420 (362 lines read)

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ L1:260 (224 lines read)

● Read appendix_v6.tex
  │ docs/paper/sections_v5/appendix_v6.tex
  └ L1:420 (420 lines read)

The current paper is much more explicit than the prior round about which counts belong to which regime, but I still need the exact released artifacts behind the “audited footprint” and “stub-mocked Verified sample” claims before I score soundness.

✗ Read README.md
  │ reproducibility/README.md
  └ Path does not exist

● Read related_v6.tex
  │ docs/paper/sections_v5/related_v6.tex
  └ L1:260 (53 lines read)

● Search (glob)
  │ "reproducibility/**/*{verified,stub,snapshot,handler_scope,footprint,headlin
  │ e,pytea,contract,audit}*"
  └ 54 files found

● Search (grep)
  │ "371|stub-mock|stub-mocked|Verified subset|handler_scope_per_block|canonical
  │ snapshot|206|208|57|55|empty-assume|26/356|15/295|15/488|Jones|Meyer|Findler
  │ |contravariant|covariant" (docs/paper/sections_v5, reproducibility, lean)
  └ 7 lines found

● Read tied_weight_stubmock_runtime.json
  │ reproducibility/tied_weight_stubmock_runtime.json
  └ 453 lines read

● Read prevalence_weighted_false_verified.json
  │ reproducibility/prevalence_weighted_false_verified.json
  └ 22 lines read

● Read pytea_mcnemar_per_bug.json
  │ reproducibility/pytea_mcnemar_per_bug.json
  └ 251 lines read

● Read reproduce_headline_60bug.json
  │ reproducibility/reproduce_headline_60bug.json
  └ 695 lines read

● Read handler_scope_per_block.json
  │ reproducibility/handler_scope_per_block.json
  └ L1:260 (260 lines read)

● Read tcb_fault_injection_footprint.json
  │ reproducibility/tcb_fault_injection_footprint.json
  └ 567 lines read

I have enough to assess the empirical claims; what remains is whether the assume/guarantee result is actually more than a framework-specific rephrasing of standard contract subtyping. I’m reading that theorem statement and its positioning now.

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ L360:560 (166 lines read)

● Read H_contribution_table.tex
  │ docs/paper/sections_v5/H_contribution_table.tex
  └ L1:260 (60 lines read)

● Read handler_soundness_table.tex
  │ docs/paper/sections_v5/handler_soundness_table.tex
  └ L1:260 (142 lines read)

● Search (grep)
  │ "sorry|ag_composition_ext|applyOpExt_sound_|lake build|theorem|lemma" (lean)
  └ 7 lines found

● Search (glob)
  │ "lean/**/*.lean"
  └ 13 files found

The mechanization claims look concrete on paper; I’m checking the released Lean artifact and the main test suite directly so the review doesn’t over-credit unverified infrastructure.

● Search (grep)
  │ "\bsorry\b" (lean)
  └ 13 lines found

● Build Lean proofs (shell)
  │ cd
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ && lake build
  └ 2 lines...

● Run pytest suite (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -m pytest tests/ -q --maxfail=1
  └ 112 lines...

One concrete artifact issue surfaced: the released test suite is not currently green because a known-shape-bug regression (`missing_unsqueeze`) now passes silently. That matters for the artifact-strength part of the review, so I’m checking whether the paper overclaims coverage for that bug family.

● Read v5_bug_corpus.jsonl
  │ experiments_v5/v5_bug_corpus.jsonl
  └ L1:120 (56 lines read)

● Read bug_corpus_manifest.json
  │ experiments_v5/bug_corpus_manifest.json
  └ L1:220 (220 lines read)

● Read test_real_model_analysis.py
  │ tests/test_real_model_analysis.py
  └ L430:500 (71 lines read)

● Search (grep)
  │ "missing unsqueeze|unsqueeze before
  │ broadcast|BUG_MISSING_UNSQUEEZE|broadcasting" (tests,
  │ docs/paper/sections_v5, experiments_v5, reproducibility)
  └ 7 lines found

I’ve got the core judgement now; I just want the exact paper wording around the bug-corpus head-to-head and the stub-mocked Verified sample so the review quotes the right section instead of paraphrasing loosely.

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L360:520 (161 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1680:1760 (81 lines read)

● Search (grep)
  │ "371|25\)|0/25|13\.32|stub-mock|stub-mocked|32/34|McNemar|15/488|5
  │ fire|62/185" (docs/paper/sections_v5/eval_v6.tex)
  └ 17 lines found

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1516:1592 (77 lines read)

● Read tied_weight_stubmock_runtime.md
  │ reproducibility/tied_weight_stubmock_runtime.md
  └ L1:180 (58 lines read)

● Read audited_footprint_unconditional_rp.md
  │ reproducibility/audited_footprint_unconditional_rp.md
  └ L1:180 (47 lines read)

## Summary
This paper presents TensorGuard, a static refinement-type verifier for PyTorch `nn.Module` forward methods that reasons about tensor shapes and a coarse gradient-flow property directly from class source, without instantiation or tracing. The technical core is a refinement calculus with assume/guarantee composition at module boundaries, plus a Lean-audited operator-rule table and a mechanized 17-operator composition fragment. Empirically, the paper claims `53/60` high-confidence refuted bugs on a historical bug corpus, a fragment-fair `32/34` vs. `25/34` win over Pytea, and `9/9` catches on naturally occurring HuggingFace-family bugs. On real-source library code, the canonical high-confidence regime reports `57` Verified, `128` Contract-Violation, `78` Library-Warn, and `225` Abstain on `488` blocks, with `0` unconditional RP on the unrestricted corpus but `26/356` on the empty-assume subset and `5` of those inside the audited footprint. The paper also calibrates its soundness claims by separating Lean-audited, pen-and-paper, tested-only, and out-of-scope handlers, and by quantifying a known backward-verifier limitation on tied / renamed-attribute parameter sharing.

## Prior weakness disposition
- [RESOLVED] The headline numerical claim in the abstract is in tension with Table `tab:headline`. The abstract advertises "26 unconditional... -- rebuttal accepted: Table `tab:headline` and the surrounding §4.1 text now explicitly distinguish unrestricted RP=`0` from the empty-`assume_M` subset count `26/356`, and the released artifacts support that reconciliation.
- [PARTIAL] C1's "joint shape-plus-grad" novelty rests on a grad lattice that is admitted to be silently incorrect on a 25% slice... -- the paper now quantifies and foregrounds the limitation, but the backward verifier still silently misverifies `2/8` positives on the worst-case tied/renamed-sharing family, so the novelty claim remains materially qualified.
- [RESOLVED] The "fragment-fair head-to-head" 32/34 vs 25/34 against Pytea is the only result with a frequentist significance... -- rebuttal accepted: the appendix now gives the full 34-row matched-pair table, the 60→34 filter rule is stated deterministically, and the released JSON reproduces the McNemar statistic.
- [RESOLVED] The "Bookkeeping note on the headline triple" ... reports four different `{V, R, A}` triples for the same 488-block corpus... -- rebuttal accepted: the current draft cleanly identifies the two axes (high-confidence vs default, original capture vs rerun), names the canonical snapshot, and the current released logs fit that bookkeeping story.
- [PARTIAL] C2 (assume/guarantee at the `nn.Module` boundary with contravariant/covariant subclassing) is, novelty-wise... -- the draft now cites Jones/Meyer/Findler and states the mechanized fragment more honestly, but the conceptual step still reads primarily as a framework-specific instantiation of standard contract subtyping.
- [PARTIAL] The "stub-mocked runtime sample on the 371-Verified subset" ... reports `0/25` silently-incorrect Verified with Wilson 95% CI... -- the authors added stronger complementary audits, but this specific sample is still only `25` rows with a `[0,13.32%]` interval and a shortest-LoC-first selection rule that overweights simple modules.
- [PARTIAL] The paper's distinctive empirical novelty — verdicts on un-instantiated class source — is most cleanly demonstrated by the inapplicability gap... -- the draft now adds `15/488`, `26/356`, and `5` audited-footprint unconditional catches, but the canonical unrestricted real-source headline still has `0/488` unconditional RP.

## Strengths
- The paper is unusually explicit about what is and is not covered by the theorem: RP/CV only, `Cat_sound` only, with tested-only and out-of-scope handlers separated rather than quietly absorbed.
- The mechanized artifact is substantive: the Lean development builds, the advertised 17-operator assume/guarantee fragment is concrete, and the operator-table audit is more serious than a typical ML-systems proof appendix.
- The empirical comparison to Pytea is now genuinely auditable; the filter, contingency table, and significance calculation are no longer black-box prose.
- The bug-finding results on historical bugs and naturally occurring HuggingFace-family bugs are strong and relevant to practice.

## Weaknesses
- The main soundness limitation remains substantial on real source: only `62/185` of the paper’s real-source Verified+CV verdicts lie wholly inside the Lean-or-pen-and-paper footprint, while `66/185` touch tested-only handlers and `57/185` touch only out-of-scope operators (`§4`, Table `tab:soundness-footprint-185`).
- The gradient-flow story is still materially weakened by the tied / renamed-attribute parameter-sharing failure mode: the runtime harness reports a `2/8 = 25%` false-Verified rate on that worst-case construct family (`§6`, `§4` runtime trainer audit).
- The stub-mocked validation on the `371` Verified tied-weight rows is not very convincing as population evidence: it samples shortest-LoC-first, succeeds on only `25` rows, and those rows are dominated by simple RMSNorm-like modules, so the reported Wilson interval `[0.00%, 13.32%]` is not tight and is selection-biased.
- The conceptual contribution around C2 still feels overstated. The theorem mechanizes composition for this DSL, but the core contravariant/covariant contract rule is standard, so the novelty seems to lie more in the PyTorch adaptation and audit packaging than in a new contract-theoretic idea.
- The paper’s most distinctive real-source claim is still weaker than the abstract framing suggests: the unrestricted `488`-block corpus yields `0` unconditional RP in the canonical regime, so the positive real-source story depends on the empty-assume subset, a rule-extension rerun, or the very small `5`-catch audited-footprint slice.
- The released artifact is not completely stable: the current test suite fails on a known bug-detection regression (`missing unsqueeze before broadcast`), which is uncomfortable for a paper whose empirical case leans heavily on a bug-catching benchmark and on implementation calibration.

## Questions
- What is the strongest real-source result that holds **strictly inside** the theorem-backed footprint, with no tested-only or out-of-scope handler anywhere on the relevant path?
- Why should C2 be read as a conceptual contribution beyond a framework-specific instantiation of standard contract/subtyping principles? What theorem obligation here is genuinely new?
- For the `371`-Verified tied-weight population, why use shortest-LoC-first rather than a stratified or random sample across handler families, and how sensitive is the `0/25` result to that selection rule?
- How should readers reconcile the current artifact regression on a broadcast/unsqueeze bug pattern with the paper’s broader bug-detection claims?

## Scores
Soundness: 2
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
I would move this to a 6 if the paper delivered a stronger **theorem-backed real-source result**, i.e. if a much larger fraction of the `488`-block Verified/CV/RP story were brought inside `Cat_sound` rather than resting on tested-only or out-of-scope handlers.

Round: 5


Changes   +0 -0
Requests  1 Premium (5m 10s)
Tokens    ↑ 952.7k • ↓ 9.4k • 826.9k (cached) • 4.3k (reasoning)
