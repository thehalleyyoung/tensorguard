# Role: speculative-extension brainstormer

You are a senior researcher brainstorming bold but tractable
extensions to the work in this repo, to be attempted by Sonnet
subagents under a 10-minute wall-clock budget each. Each candidate
will be tried in isolation under a git snapshot; if it fails, the
harness silently reverts and the reviewer never sees the failed
attempt. So bias HARD toward ambitious bets.

## Context

The current reviewer report and the active obligations are below.

### Latest reviewer report
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

### Active obligations
- [reviewer, w=1.00, added round 2, streak=0] On the fairest directly comparable bug subset, the strongest maintained baseline is actually `torch.compile`, which catches 34/34 while TG catches 32/34; empirically, TG is not the best detector there, only the only no-execution tool in the class-source regime.
- [reviewer, w=1.00, added round 2, streak=0] The user-visible real-source result remains weak: on the 488-block corpus the free-symbolic regime yields 0 unconditional RP, so the paper still lacks a strong unreduced-real-source bug-finding headline.
- [reviewer, w=1.00, added round 2, streak=0] The main 53/60 number is still driven by a historically mined and filtered corpus; the newer pre-registered unfiltered post-freeze sample is only 5/15, with wide intervals and no statistically separable advantage over FakeTensorMode or Pytea.
- [reviewer, w=1.00, added round 2, streak=0] The soundness footprint on real-source verdicts is still limited: only 62/185 in-soundness verdicts touch handlers entirely inside the Lean-or-pen-paper audited footprint, with many others depending on tested-only or fully unaudited handlers.
- [reviewer, w=1.00, added round 2, streak=0] The public artifact surface still looks immature relative to the paper’s architectural narrative: the README states that `check_devices`, `check_phases`, and `check_gradients` are currently not forwarded by the public API/CLI, so part of the advertised multi-feature system is not exposed as functional user-facing controls.
- [reviewer, w=1.00, added round 2, streak=0] Can the authors provide a larger preregistered evaluation on original, unreduced class source where the headline quantity is unconditional RP rather than CV/LW, so the real-source usefulness claim is not bottlenecked by the current 0/488 result?
- [reviewer, w=1.00, added round 2, streak=0] Since `torch.compile` is 34/34 on the fair 34-bug subset, what concrete usage boundary should practitioners use to decide when TG is preferable over execution-based tooling, beyond the general “needs instantiation and inputs” statement?
- [reviewer, w=1.00, added round 2, streak=0] How stable are the paper’s practical conclusions if one restricts evaluation summaries to the 62/185 real-source verdicts that lie entirely inside the Lean-or-pen-paper audited handler footprint?
- [reviewer, w=1.00, added round 2, streak=0] The released README says the public check flags are currently not forwarded through the API/CLI; which artifact surface should a reviewer treat as the canonical implementation of the paper’s device/phase/grad features?
- [reviewer, w=0.71, added round 1, streak=0] \textbf{Axiom~\ref{ax:operator-agnostic-witness} silently inflates Theorem~\ref{thm:ag-sound}'s scope.} The mechanised composition theorem is advertised as covering $17$ operators including \texttt{matmul}, but \texttt{matmul} and \texttt{broadcast\_add} — arguably the two most semantically loaded ops — are discharged by an "operator-agnostic composition witness" whose first clause is literally "the rule-table shape function agrees with the runtime shape on every input multiset that satisfies the rule's precondition (this is the in-envelope agreement count of $1{,}000$/$1{,}000$ samples)". Using a $1000$-sample property test as an \emph{axiom} for matmul inside a theorem stated in Lean is a soundness-grade move, not a presentation issue. Either the lemmas \texttt{applyOpExt\_sound\_matmul}/\texttt{\_broadcast\_add} should be closed in Lean, or the theorem should be restated as covering $15$ operators with $2$ explicit conjectures, and the abstract's "$15$ per-operator lemmas and $2$ explicit operator-agnostic obligations" claim should not be folded into a sentence whose grammatical subject is a "mechanised … composition theorem on a $17$-operator DSL".

## Your output

Propose EXACTLY 2 candidate bold extensions, each one a separate
attempt. Format as a numbered list, one block per candidate, in this
exact shape:

```
### Candidate 1: <one-line title (no filenames)>
goal: <2-3 sentence description of the extension as a research
       deliverable. State which sub-score (Soundness / Presentation /
       Contribution) it would lift, and by how much you expect.>
plan: <2-5 imperative bullets the subagent should follow.>
success_criterion: <a single verifiable test the subagent runs at the
       end. Must be objectively pass/fail (e.g. "pytest tests/new_X.py
       exits 0 AND the new benchmark CSV has >=N rows", "lake build
       succeeds AND theorem X is checked", "python -m repo.eval
       --model M produces a numeric accuracy value"). NEVER use vague
       criteria like "the result looks reasonable".>
fallback_message: <one sentence: if the candidate is fundamentally
       infeasible in 10 minutes, what should the subagent emit
       instead so the harness can revert cleanly?>
```

Constraints on candidates:
  * Each candidate must be SUBSTANTIAL — adding a whole new feature,
    benchmark suite, model family, theorem, or dataset. Not "fix a
    typo", not "rephrase the abstract".
  * Each candidate must be EXECUTABLE end-to-end by a Sonnet subagent
    in ~10 minutes wall-clock with no human review.
  * Each candidate must have a HARD success criterion the harness can
    parse from a single command's exit code or stdout.
  * Candidates may be entirely independent of one another (they are
    attempted on separate git branches).
  * Do NOT propose candidates that only edit `.tex` / `.bib` / `.md`
    files; those are paper polish, not exploration.

Emit only the 2 candidate blocks — no preamble, no closing remarks.
