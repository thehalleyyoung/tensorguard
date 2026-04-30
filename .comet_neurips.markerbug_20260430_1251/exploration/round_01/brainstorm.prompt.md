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
The paper presents \textsc{TensorGuard}, a refinement-type checker for PyTorch \texttt{nn.Module} class \emph{source} that statically computes a joint shape-and-grad type \(\mathsf{Tensor}\{s,g\mid\varphi\}\), with Z3 discharging shape obligations and a flat three-point grad lattice tracking autograd-tape liveness. Verdicts are partitioned into Verified, Refuted-Proof (RP), Contract-Violation (CV), Library-Warn (LW), and Abstain; the soundness theorem covers RP and CV only, and only over the 44-of-79-handler sub-catalogue \(\mathrm{Cat}_{\mathrm{sound}}\). The empirical claims include 53/60 RP on a curated historical bug corpus, 32/34 vs.\ Pytea 25/34 (McNemar exact \(p{=}0.0156\)) on a fragment-fair head-to-head, 9/9 on transcribed cross-family HuggingFace bug-fix PRs, and a Lean~4 development with 11/11 previously-axiomatic soundness lemmas closed sorry-free plus an assume/guarantee composition theorem on a 17-operator DSL. An exploratory necessary-direction Dynamo-guard inclusion lemma is supplied on a 17-module audit and is explicitly not claimed as runtime equivalence.

## Prior weakness disposition
(none — first round)

## Strengths
- The verdict taxonomy is unusually disciplined: the soundness theorem is explicitly scoped to RP+CV on \(\mathrm{Cat}_{\mathrm{sound}}\), the 35 tested-only handlers are demoted to a conjecture (\Cref{conj:tested-only-soundness}), and the Dynamo correspondence is labeled exploratory and one-directional. This is the right epistemic shape for a verification paper.
- A genuine Preservation/Progress development for \(\mathcal{F}_{\mathrm{TG}}\) is supplied in the appendix (\texttt{subject\_reduction\_v8.tex}), with operational semantics, a Reduction Lemma case analysis, and a heap-satisfaction definition that ties the symbolic refinements to concrete runtime values.
- The Lean library targets (\texttt{TensorGuard.*}) do build with grep showing no `sorry` tokens in the substantive operator-rule files (\texttt{V5OperatorRules}, \texttt{Soundness}, \texttt{AssumeGuarantee}, \texttt{Extended}, \texttt{Parity}); residual `sorry` mentions are inside doc-strings, not proof obligations.
- The Pytea head-to-head is matched-pair clean: \(b{=}7\), \(c{=}0\) on the 34-row fragment-fair subset, so the reported gap is not a small-\(N\) artifact in the asymmetric direction (Pytea-refutes \(\subseteq\) TG-refutes).
- The paper avoids the usual self-serving headline by openly stating that the user-visible free-symbolic regime produces 0 unconditional RP on 488 blocks; the unconditional-RP story rests on the bug corpora, not the block corpus.

## Weaknesses
- **`lake build` is not green.** \texttt{lean/build\_round2.log} shows \texttt{ParityRunner} failing with a Lean type error (`Array.map (fun n => Json.num n) ns.toArray` — expected `Array Lean.JsonNumber`, got `Array Nat`), and the build exits with `error: build failed`. The library `.olean`s do build, but the abstract's claim that "the operator-rule tree building \texttt{sorry}-free under \texttt{lake build}" is materially misleading: under `lake build` the project itself does not build. A reviewer cannot distinguish "the operator-rule library compiles" from "the Lean project is healthy" without auditing the log. Either fix `ParityRunner.lean` or scope the claim to `lake build TensorGuard.V5OperatorRules` (etc.).
- **Theorem \ref{thm:soundness}(ii) is overclaimed relative to what is proved.** Case (ii) asserts that on RP, *every* reduction sequence raises a shape- or grad-flag exception at the witnessed subterm. The proof sketch in \texttt{calculus\_v6.tex} L132–146 reduces (i) and (ii) to per-operator *preservation* lemmas. Preservation gives soundness of Verified, not completeness of RP. To get (ii) you additionally need: (a) the witness produced by Z3 satisfies \(\Gamma\) and the path predicate, (b) the operator's runtime semantics actually raises on that witness (not merely is undefined in the rule), and (c) reduction is deterministic/confluent enough that "every" sequence reaches the witnessed site. None of these is explicit in the appendix, and the Reduction Lemma covers preservation only. Either weaken (ii) to "there exists a heap \(\sigma\models\Gamma\) such that ..." or supply the missing completeness argument.
- **Axiom \ref{ax:fresh-witness} is a property of the Python implementation, not a property of the calculus, and the paper grants it axiom status to derive Thm.\ \ref{thm:monotonicity} (Monotonicity).** A theorist reading this is being asked to take on faith that the deployed analyser does not memoise witnesses across passes — an implementation invariant that is not under Lean and not under any property test cited in the paper. Either mechanise the no-memoisation property against the analyser, or restate Monotonicity as a conditional ("under the no-memoisation discipline ...").
- **The pen-and-paper handlers are load-bearing for \(\mathrm{Cat}_{\mathrm{sound}}\) but their soundness arguments are extremely terse.** \texttt{handler\_soundness\_table.tex} L40–50 includes \texttt{einsum}, \texttt{elementwise\_binary}, \texttt{reduce}, \texttt{dropout}, \texttt{pad}, \texttt{where}, \texttt{softmax}, \texttt{flatten}, etc. These are nontrivial: \texttt{einsum} alone has subscript-arity, output-subscript-uniqueness, and broadcast-axis preconditions; \texttt{pad} interacts with the rank arithmetic; \texttt{where} interacts with the grad lattice. The appendix's L314–318 dispatches all three of \texttt{T-Broadcast/T-Reduce/T-Einsum} as "verified by the same argument as \textsc{T-Cat}", which is not credible for \texttt{einsum}. For a "proof-grade" sub-catalogue this is below the bar.
- **The \(\le 12\%\) prevalence ceiling on the parameter-sharing-under-renamed-attribute silent-incorrectness is a regex-detectable bound, not a semantic one,** and the paper is explicit about that (\texttt{limconc\_v6.tex} L132–138). Yet the same paragraph reports a worst-case false-verified rate of 2/8 \(=\) 25.0\% on the targeted construct family. The headline grad-flag claims (8/8 caught, 0/50 false positives, 500/500 agreement) live in \texttt{impl\_v6.tex} §3.2 and never appear next to the 25\% worst-case number. As a soundness reader I have no way to compose these into a single bound on the false-Verified rate of the backward verifier in deployment, because the regex bound and the worst-case bound are over different populations and the prevalence-weighted product is not given. Please supply the prevalence-weighted worst-case false-Verified estimate, or weaken C3.
- **\(\mathrm{Cat}_{\mathrm{tested}}\) (35/79 handlers, \(\sim\)44%) is supported only by 1{,}000-sample property tests (\Cref{conj:tested-only-soundness}).** A 1000-sample empirical agreement against PyTorch is not a soundness argument in any meaningful sense for handlers that take symbolic shapes — the property test concretises before sampling, so it cannot exhibit any disagreement that occurs only at unrealisable concrete sizes, broadcast-collision corner cases, or symbolic divisibility witnesses. The paper correctly does not call this a theorem, but the headline "79-handler catalogue" framing in C2 papers over the fact that \(\sim\)44% of the deployed catalogue carries no soundness argument at all.
- **The \(\mathrm{Cat}_{\mathrm{sound}}\)-witnessed real-source RP count is 5 (out of 488 blocks), per the abstract.** This is the only real-source unconditional refutation count covered by Thm.\ \ref{thm:soundness}. Five witnesses on 488 blocks is a thin empirical anchor for a paper whose theoretical contribution is precisely the audited footprint; the 53/60 historical-corpus headline rests almost entirely on handlers in \(\mathrm{Cat}_{\mathrm{tested}}\) (no decomposition is given), so a theorist cannot tell what fraction of the bug-corpus RPs are inside the soundness theorem.

## Questions
- Please decompose the 53/60 historical-corpus RP count by which sub-catalogue the firing handler chain lies in: how many fire entirely inside \(\mathrm{Cat}_{\mathrm{audit}}\), entirely inside \(\mathrm{Cat}_{\mathrm{sound}}\), or touch \(\mathrm{Cat}_{\mathrm{tested}}\)? Without this split the headline is not a soundness claim.
- For Thm.\ \ref{thm:soundness}(ii), supply the explicit completeness step that goes from "Z3 SAT on the negated obligation produces a model" to "every reduction sequence raises an exception at the witnessed subterm". In particular, please address determinism/confluence of the small-step semantics on inputs satisfying the witness, and the model-extraction step that turns a Z3 model into a heap \(\sigma\models\Gamma\).
- The Lean assume/guarantee theorem is mechanised on a 17-operator DSL with 15/17 per-operator \texttt{applyOpExt\_sound\_*} lemmas; the remaining two (\texttt{broadcast\_add}, \texttt{matmul}) are described as "fall through to the operator-agnostic composition witness". Could you state precisely what "operator-agnostic composition witness" hypothesises about these two operators — i.e., what unproved obligation is being moved into the trusted base?
- Why does \texttt{lake build} fail on \texttt{ParityRunner}? Is \texttt{ParityRunner} part of the cited "operator-rule tree" or not? If not, please isolate the artifact reviewers are expected to build (e.g., a `lake build TensorGuard` target that excludes the runner).
- For the pen-and-paper \texttt{einsum} handler, please give the precise statement of soundness, including the side-conditions on subscript repetition and broadcast-axis identification. Citing "the same argument as \textsc{T-Cat}" is insufficient.
- Axiom \ref{ax:fresh-witness}: can you give an executable test (or a syntactic invariant on the Python source) that any reviewer could run against the implementation to verify the no-memoisation discipline?

## Scores
Soundness: 2
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would move me from 5 to 6 is fixing the \texttt{lake build} green-build claim end-to-end and supplying the explicit completeness argument for Theorem \ref{thm:soundness}(ii) (or weakening the statement to existential form). Both are mechanical fixes that would close the most serious soundness gaps; absent them, the paper is borderline-reject by the rigor bar a theorist applies.


Changes   +0 -0
Requests  7.5 Premium (2m 49s)
Tokens    ↑ 1.3m • ↓ 8.0k • 1.2m (cached)

### Active obligations
- [reviewer, w=1.00, added round 1, streak=0] **`lake build` is not green.** \texttt{lean/build\_round2.log} shows \texttt{ParityRunner} failing with a Lean type error (`Array.map (fun n => Json.num n) ns.toArray` — expected `Array Lean.JsonNumber`, got `Array Nat`), and the build exits with `error: build failed`. The library `.olean`s do build, but the abstract's claim that "the operator-rule tree building \texttt{sorry}-free under \texttt{lake build}" is materially misleading: under `lake build` the project itself does not build. A reviewer cannot distinguish "the operator-rule library compiles" from "the Lean project is healthy" without auditing the log. Either fix `ParityRunner.lean` or scope the claim to `lake build TensorGuard.V5OperatorRules` (etc.).
- [reviewer, w=1.00, added round 1, streak=0] **Theorem \ref{thm:soundness}(ii) is overclaimed relative to what is proved.** Case (ii) asserts that on RP, *every* reduction sequence raises a shape- or grad-flag exception at the witnessed subterm. The proof sketch in \texttt{calculus\_v6.tex} L132–146 reduces (i) and (ii) to per-operator *preservation* lemmas. Preservation gives soundness of Verified, not completeness of RP. To get (ii) you additionally need: (a) the witness produced by Z3 satisfies \(\Gamma\) and the path predicate, (b) the operator's runtime semantics actually raises on that witness (not merely is undefined in the rule), and (c) reduction is deterministic/confluent enough that "every" sequence reaches the witnessed site. None of these is explicit in the appendix, and the Reduction Lemma covers preservation only. Either weaken (ii) to "there exists a heap \(\sigma\models\Gamma\) such that ..." or supply the missing completeness argument.
- [reviewer, w=1.00, added round 1, streak=0] **Axiom \ref{ax:fresh-witness} is a property of the Python implementation, not a property of the calculus, and the paper grants it axiom status to derive Thm.\ \ref{thm:monotonicity} (Monotonicity).** A theorist reading this is being asked to take on faith that the deployed analyser does not memoise witnesses across passes — an implementation invariant that is not under Lean and not under any property test cited in the paper. Either mechanise the no-memoisation property against the analyser, or restate Monotonicity as a conditional ("under the no-memoisation discipline ...").
- [reviewer, w=1.00, added round 1, streak=0] **The pen-and-paper handlers are load-bearing for \(\mathrm{Cat}_{\mathrm{sound}}\) but their soundness arguments are extremely terse.** \texttt{handler\_soundness\_table.tex} L40–50 includes \texttt{einsum}, \texttt{elementwise\_binary}, \texttt{reduce}, \texttt{dropout}, \texttt{pad}, \texttt{where}, \texttt{softmax}, \texttt{flatten}, etc. These are nontrivial: \texttt{einsum} alone has subscript-arity, output-subscript-uniqueness, and broadcast-axis preconditions; \texttt{pad} interacts with the rank arithmetic; \texttt{where} interacts with the grad lattice. The appendix's L314–318 dispatches all three of \texttt{T-Broadcast/T-Reduce/T-Einsum} as "verified by the same argument as \textsc{T-Cat}", which is not credible for \texttt{einsum}. For a "proof-grade" sub-catalogue this is below the bar.
- [reviewer, w=1.00, added round 1, streak=0] **The \(\le 12\%\) prevalence ceiling on the parameter-sharing-under-renamed-attribute silent-incorrectness is a regex-detectable bound, not a semantic one,** and the paper is explicit about that (\texttt{limconc\_v6.tex} L132–138). Yet the same paragraph reports a worst-case false-verified rate of 2/8 \(=\) 25.0\% on the targeted construct family. The headline grad-flag claims (8/8 caught, 0/50 false positives, 500/500 agreement) live in \texttt{impl\_v6.tex} §3.2 and never appear next to the 25\% worst-case number. As a soundness reader I have no way to compose these into a single bound on the false-Verified rate of the backward verifier in deployment, because the regex bound and the worst-case bound are over different populations and the prevalence-weighted product is not given. Please supply the prevalence-weighted worst-case false-Verified estimate, or weaken C3.
- [reviewer, w=1.00, added round 1, streak=0] **\(\mathrm{Cat}_{\mathrm{tested}}\) (35/79 handlers, \(\sim\)44%) is supported only by 1{,}000-sample property tests (\Cref{conj:tested-only-soundness}).** A 1000-sample empirical agreement against PyTorch is not a soundness argument in any meaningful sense for handlers that take symbolic shapes — the property test concretises before sampling, so it cannot exhibit any disagreement that occurs only at unrealisable concrete sizes, broadcast-collision corner cases, or symbolic divisibility witnesses. The paper correctly does not call this a theorem, but the headline "79-handler catalogue" framing in C2 papers over the fact that \(\sim\)44% of the deployed catalogue carries no soundness argument at all.
- [reviewer, w=1.00, added round 1, streak=0] **The \(\mathrm{Cat}_{\mathrm{sound}}\)-witnessed real-source RP count is 5 (out of 488 blocks), per the abstract.** This is the only real-source unconditional refutation count covered by Thm.\ \ref{thm:soundness}. Five witnesses on 488 blocks is a thin empirical anchor for a paper whose theoretical contribution is precisely the audited footprint; the 53/60 historical-corpus headline rests almost entirely on handlers in \(\mathrm{Cat}_{\mathrm{tested}}\) (no decomposition is given), so a theorist cannot tell what fraction of the bug-corpus RPs are inside the soundness theorem.
- [reviewer, w=1.00, added round 1, streak=0] Please decompose the 53/60 historical-corpus RP count by which sub-catalogue the firing handler chain lies in: how many fire entirely inside \(\mathrm{Cat}_{\mathrm{audit}}\), entirely inside \(\mathrm{Cat}_{\mathrm{sound}}\), or touch \(\mathrm{Cat}_{\mathrm{tested}}\)? Without this split the headline is not a soundness claim.
- [reviewer, w=1.00, added round 1, streak=0] For Thm.\ \ref{thm:soundness}(ii), supply the explicit completeness step that goes from "Z3 SAT on the negated obligation produces a model" to "every reduction sequence raises an exception at the witnessed subterm". In particular, please address determinism/confluence of the small-step semantics on inputs satisfying the witness, and the model-extraction step that turns a Z3 model into a heap \(\sigma\models\Gamma\).
- [reviewer, w=1.00, added round 1, streak=0] The Lean assume/guarantee theorem is mechanised on a 17-operator DSL with 15/17 per-operator \texttt{applyOpExt\_sound\_*} lemmas; the remaining two (\texttt{broadcast\_add}, \texttt{matmul}) are described as "fall through to the operator-agnostic composition witness". Could you state precisely what "operator-agnostic composition witness" hypothesises about these two operators — i.e., what unproved obligation is being moved into the trusted base?

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
