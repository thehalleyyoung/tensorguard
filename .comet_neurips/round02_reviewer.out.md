● Read round02_reviewer.md
  │ .comet_neurips/round02_reviewer.md
  └ 284 lines read

I’m checking the specific claims the review prompt highlights against the current paper text and released artifact so the dispositions and scores line up with the actual repo evidence.

● Read real_benchmark_results.json
  │ benchmarks/real_benchmark_results.json
  └ 1 line read

● Read dl4c_bench_results.json
  │ benchmarks/dl4c_bench_results.json
  └ 101 lines read

● Read headline_results.json
  │ benchmarks/headline_results.json
  └ 28 lines read

● Read pyproject.toml
  └ 44 lines read

● Read lean_parity_results.json
  │ benchmarks/lean_parity_results.json
  └ 13 lines read

● Search (grep)
  │ "ax:operator-agnostic-witness|thm:ag-sound|def:model-extraction|thm:soundnes
  │ s|rem:tcb-thm-ii|ax:fresh-witness|thm:monotonicity|53/60|488|34/34|32/34|Pyt
  │ ea|torch\.compile|Wilson|CI \[|Axiom|Theorem|Preservation|Progress" in
  │ *.{tex,txt}
  └ 8 lines found

● Read copilot-tool-output-1777570044864-msbkh6.txt
  │ /var/folders/2k/n00k2r_d7sb3g3g0jkf059mh0000gn/T/copilot-tool-output-1777570
  │ 044864-msbkh6.txt
  └ L1:220 (176 lines read)

● Read headline_results.json
  │ benchmarks/headline_results.json
  └ L1:220 (28 lines read)

● Search (glob)
  │ "**/*soundness*.tex"
  └ 2 files found

● Search (glob)
  │ "**/main.tex"
  └ No matches found

● Search (glob)
  │ "**/neurips.tex"
  └ 1 file found

● Read neurips.txt
  └ L330:455 (125 lines read)

● Read neurips.txt
  └ L690:725 (36 lines read)

● Read neurips.txt
  └ L1688:1712 (25 lines read)

● Read neurips.txt
  └ L1058:1210 (153 lines read)

● Read neurips.txt
  └ L3278:3332 (55 lines read)

I found the paper has materially strengthened several theory sections, but the empirical story still has a real asymmetry: the strongest current baseline in the comparable bug subset is actually `torch.compile`, and the real-source headline still has zero unconditional proof-grade catches.

✗ Search (grep)
  │ "model extraction|requires_grad|grad component|three-element|def:model-extra
  │ ction|m\(g_i\)|\mathfrak\{m\}\(g_i\)|fresh-witness|operator-agnostic
  │ composition witness|applyOp sound|Axiom" in *.txt
  └ rg: regex parse error:
    (?:model extraction|requires_grad|grad component|three-element|def:model-ext
    raction|m\(g_i\)|\mathfrak\{m\}\(g_i\)|fresh-witness|operator-agnostic
    composition witness|applyOp sound|Axiom)
    ^^
    error: unrecognized escape sequence

● Search (glob)
  │ "**/*pytea*"
  └ 57 files found

● Search (glob)
  │ "**/*real*benchmark*"
  └ 11 files found

● Search (glob)
  │ "**/*ablation*"
  └ 22 files found

● Search (grep)
  │ "53/60|32/34|34/34|118/128|488-block|0-RP|free
  │ symbolic|mutation-testing|seeds|seed|error bars|confidence
  │ interval|Clopper|Wilson|bootstrap|McNemar" in *.json
  └ 5 lines found

● Read pytea_fragment_fair.json
  │ reproducibility/pytea_fragment_fair.json
  └ L1:200 (200 lines read)

● Read per_rule_ablation_60bug.json
  │ reproducibility/per_rule_ablation_60bug.json
  └ L1:220 (220 lines read)

● Read README.md
  └ L1:220 (220 lines read)

● Read pytea_mcnemar_per_bug.json
  │ reproducibility/pytea_mcnemar_per_bug.json
  └ L1:120 (120 lines read)

● Read real_corpus_ablation.json
  │ reproducibility/real_corpus_ablation.json
  └ L1:200 (200 lines read)

The repo evidence also exposes a stronger artifact-level concern: the shipped CLI documents `--no-phase-check`, `--no-device-check`, and `--no-grad-check` as current no-ops, so some of the multi-theory implementation claims are not yet reflected in the public tool surface.

● Read feature_ablation.json
  │ experiments_v5/feature_ablation.json
  └ L1:220 (175 lines read)

● Read real_corpus_ablation.md
  │ reproducibility/real_corpus_ablation.md
  └ L1:220 (31 lines read)

● Read fp_ablation_results.json
  │ benchmarks/fp_ablation_results.json
  └ L1:220 (220 lines read)

● Search (grep)
  │ "phase|device|grad|5-theory|five-theory|no-phase-check|no-device-check|no-gr
  │ ad-check|phase errors|device inconsistencies|gradient" in *.txt
  └ 8 lines found

● Read neurips.txt
  └ L230:315 (86 lines read)

● Read copilot-tool-output-1777570088351-d4797b.txt
  │ /var/folders/2k/n00k2r_d7sb3g3g0jkf059mh0000gn/T/copilot-tool-output-1777570
  │ 088351-d4797b.txt
  └ L1:120 (120 lines read)

● Search (grep)
  │ "model extraction|requires_grad|m\(g_i\)|three-element|Definition 1|Def\.
  │ 1|Def\. 2|def:model-extraction" in neurips.txt
  └ No matches found

● Read paper_check.txt
  └ L240:290 (51 lines read)

● Read paper_check.txt
  └ L1760:1885 (126 lines read)

● Search (grep)
  │ "Theorem 5|necessary
  │ direction|operator-agnostic|fresh-witness|Axiom|matmul|17-operator|classical
  │ Preservation|Progress" in paper_check.txt
  └ 41 lines found

● Read paper_check.txt
  └ L210:240 (30 lines read)

● Read paper_check.txt
  └ L430:550 (121 lines read)

● Search (grep)
  │ "model extraction|requires grad|three-element|Theorem 2
  │ \(Soundness|Definition|m\(g_i\)|\{has grad, no grad, ⊤\}|grad lattice is
  │ flat" in paper_check.txt
  └ 21 lines found

● Read paper_check.txt
  └ L1:190 (190 lines read)

● Search (grep)
  │ "reproducibility appendix|reproducibility|command|generated_by|python3
  │ reproducibility|catalogue freeze" in paper_check.txt
  └ 34 lines found

● Search (grep)
  │ "53/60|11/57|128|78|57|488|60
  │ historical|public-repo|torch\.compile|FakeTensor|7/10|8/10|zero
  │ unconditional" in paper_check.txt
  └ 120 lines found

I’ve got enough evidence to score this as a materially improved but still empirically mixed paper: the formal scoping issues are mostly cleaned up, while the main remaining problems are on the experimental side and on the public artifact surface.

● Read paper_check.txt
  └ L1138:1172 (35 lines read)

● Read paper_check.txt
  └ L1258:1435 (178 lines read)

● Read paper_check.txt
  └ L1498:1565 (68 lines read)

## Summary
This paper presents TensorGuard, a no-execution static verifier for PyTorch `nn.Module` class source that uses refinement types plus Z3 to reason about tensor shapes and a first-order gradient-flow flag without instantiating the model or tracing it. The formal contribution is a typed calculus for the supported fragment, an assume/guarantee composition rule at module boundaries, and a Lean-audited subset of the operator-rule table, with additional pen-and-paper proofs and explicitly scoped tested-only handlers. Empirically, the paper reports 53/60 proof-grade catches on a curated historical bug corpus, a fragment-fair 32/34 vs. 25/34 comparison against Pytea, and zero unconditional proof-grade catches on the 488-block real-source corpus in the free-symbolic regime. The revision also adds stronger calibration machinery: explicit theorem-to-TCB scoping, per-bug contingency tables, feature ablations, a 10-bug upstream-faithful real-bug set, and a pre-registered post-freeze sample. The central claim is therefore not “best raw bug-finding everywhere,” but rather “soundly calibrated static reasoning from unreduced class source in a regime where execution-based tools are often mechanically inapplicable.”

## Prior weakness disposition
- [RESOLVED] **Axiom~\ref{ax:operator-agnostic-witness} silently inflates Theorem~\ref{thm:ag-sound}'s scope.** -- The current theorem statement now explicitly says the mechanised fragment is 17 operators, with 15 carrying per-operator lemmas and matmul/broadcast-add handled by the operator-agnostic composition witness, so the scope inflation is no longer silent.
- [RESOLVED] **The model-extraction definition (\Cref{def:model-extraction}) is mathematically incomplete on the grad component.** -- The paper now explicitly defines the grad abstraction as the flat lattice `{has grad, no grad, ⊤}` over `requires_grad ∧ is on tape`, so the missing mathematical object is supplied.
- [PARTIAL] **The "soundness theorem" of \Cref{thm:soundness} is conditional on three load-bearing TCB obligations enumerated in \Cref{rem:tcb-thm-ii}...** -- Rebuttal rejected: the revision clearly exposes the TCB obligations and their empirical support, but the handler-raises-on-`¬φ_op` step still rests on documentation plus agreement tables rather than a derivation from the implementation.
- [RESOLVED] **Axiom~\ref{ax:fresh-witness} (fresh-witness refutation) is an axiom about the implementation, not the calculus.** -- Rebuttal accepted: the paper now labels this explicitly as an implementation axiom and states monotonicity only under that hypothesis, which fixes the previous smuggling problem.
- [PARTIAL] **The headline empirical claims rest on heavily curated corpora.** -- The addition of the 10-bug upstream-faithful set and the pre-registered 15-bug post-freeze sample is meaningful progress, but the main 53/60 headline still comes from a historically mined and filtered corpus.
- [RESOLVED] **Pytea baseline is essentially abandoned.** -- Rebuttal accepted: the paper now gives a dedicated `torch.compile` comparison showing 34/34 on the same 34 bugs and uses Pytea only as the closest same-regime static baseline, so the framing problem is substantially fixed.
- [RESOLVED] **The relationship between Theorem~\ref{thm:soundness} and the classical Preservation/Progress pair is asserted, not derived.** -- The current draft now states the classical theorems separately and explicitly derives the verdict theorem from them, rather than merely gesturing at the connection.

## Strengths
- The empirical presentation is much stronger than a typical systems-for-ML submission: confidence intervals, matched-pair tests, per-bug contingency tables, and reproducibility artifacts make the headline numbers auditable.
- The paper is now substantially better calibrated about scope: it distinguishes Lean-audited, pen-and-paper, tested-only, and outside-scope handlers instead of letting “mechanised” overclaim the theorem.
- The fragment-fair Pytea comparison is well executed; the 32/34 vs. 25/34 result is backed by a released membership predicate and per-item table rather than informal filtering.
- The added upstream-faithful and post-freeze evaluations are valuable because they move at least part of the empirical story away from the original curated 60-bug corpus.
- The paper is unusually honest about negative results: zero unconditional RP on the 488-block real-source corpus, no-op knobs, and regime asymmetries are now surfaced rather than hidden.

## Weaknesses
- On the fairest directly comparable bug subset, the strongest maintained baseline is actually `torch.compile`, which catches 34/34 while TG catches 32/34; empirically, TG is not the best detector there, only the only no-execution tool in the class-source regime.
- The user-visible real-source result remains weak: on the 488-block corpus the free-symbolic regime yields 0 unconditional RP, so the paper still lacks a strong unreduced-real-source bug-finding headline.
- The main 53/60 number is still driven by a historically mined and filtered corpus; the newer pre-registered unfiltered post-freeze sample is only 5/15, with wide intervals and no statistically separable advantage over FakeTensorMode or Pytea.
- The soundness footprint on real-source verdicts is still limited: only 62/185 in-soundness verdicts touch handlers entirely inside the Lean-or-pen-paper audited footprint, with many others depending on tested-only or fully unaudited handlers.
- The public artifact surface still looks immature relative to the paper’s architectural narrative: the README states that `check_devices`, `check_phases`, and `check_gradients` are currently not forwarded by the public API/CLI, so part of the advertised multi-feature system is not exposed as functional user-facing controls.

## Questions
- Can the authors provide a larger preregistered evaluation on original, unreduced class source where the headline quantity is unconditional RP rather than CV/LW, so the real-source usefulness claim is not bottlenecked by the current 0/488 result?
- Since `torch.compile` is 34/34 on the fair 34-bug subset, what concrete usage boundary should practitioners use to decide when TG is preferable over execution-based tooling, beyond the general “needs instantiation and inputs” statement?
- How stable are the paper’s practical conclusions if one restricts evaluation summaries to the 62/185 real-source verdicts that lie entirely inside the Lean-or-pen-paper audited handler footprint?
- The released README says the public check flags are currently not forwarded through the API/CLI; which artifact surface should a reviewer treat as the canonical implementation of the paper’s device/phase/grad features?

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons
A single result showing **materially nonzero unconditional proof-grade catches on unreduced real-source class code** would raise my score by one point. Right now the paper is rigorous and much better calibrated than before, but its strongest practical claim is still bottlenecked by the 0/488 real-source unconditional result.

Round: 2


Changes   +0 -0
Requests  1 Premium (2m 58s)
Tokens    ↑ 652.9k • ↓ 8.6k • 574.6k (cached) • 4.1k (reasoning)
