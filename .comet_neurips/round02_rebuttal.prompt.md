# Role: paper authors writing a brief rebuttal

A NeurIPS reviewer just posted the review below on your paper. Before
you start any code or paper changes this round, you have ONE chance to
push back on weaknesses you believe are misweighted, factually wrong,
or already-resolved-in-the-current-repo. The next round's reviewer
WILL read this rebuttal and must either accept it (drop the weakness)
or sharpen it (restate with a concrete counter-example).

## The review you are rebutting
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

## Output requirements

Pick **at most 3** of the listed weaknesses. For each, write a
paragraph of strict format:

  ### Rebuttal of weakness: <verbatim wording, truncated to ~100 chars>
  Concise argument (4-8 sentences) for why this weakness is
  overweighted, factually wrong, or already addressed. Cite specific
  artifacts in the repo (concept names, theorem names, table numbers
  — NOT file paths) that prove your point. Do NOT add caveats. Do
  NOT use the word "honest" or any rebuttal-style narration that
  mentions the reviewer.

If you have nothing strong enough to rebut, write only the line:
`(no rebuttal this round — addressing all weaknesses in the improver pass)`

Do not preface with anything; the first non-blank line of your output
must be either the first `### Rebuttal of weakness:` header or the
`(no rebuttal this round...)` sentinel. Do not write to a file.

Round: 2
