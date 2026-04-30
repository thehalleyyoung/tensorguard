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

The paper presents \textsc{TensorGuard}, a static refinement-type checker for PyTorch \texttt{nn.Module} forward methods that works directly from class source (no instantiation, no tracing). The type system pairs symbolic shape predicates with a flat grad-flow lattice, discharged via Z3 over a LIA + divisibility + bounded-nonlinear-multiplication fragment. The authors give a five-verdict taxonomy (V/RP/CV/LW/Abstain), state a soundness theorem restricted to a 44-handler sub-catalogue $\mathrm{Cat}_{\mathrm{sound}}$ (28 Lean-mechanised + 16 pen-and-paper), and a Lean-mechanised assume/guarantee composition theorem on a 17-operator DSL. Empirically they report $53/60$ Refuted-Proof on a curated historical bug corpus, $32/34$ vs.\ Pytea $25/34$ on a fragment-fair subset (McNemar $p=0.0156$), $9/9$ on transcribed cross-family transformer bugs, and $0$ unconditional RP on the $488$-block real-source corpus.

## Prior weakness disposition
(none — first round)

## Strengths
- The paper is unusually disciplined about \emph{what} is mechanised: it explicitly partitions the 79-handler catalogue into audited / pen-and-paper / tested-only sub-catalogues, scopes Theorem~\ref{thm:soundness} to $\mathrm{Cat}_{\mathrm{sound}}$ only, and reports verdicts split by which sub-catalogue each handler chain touches. This is exactly the calibration discipline a theorist wants.
- The Lean development is real and (verified locally) builds sorry-free: `grep -nE "(:= sorry|by sorry|^[[:space:]]*sorry$)" lean/TensorGuard/*.lean lean/TheoryCombination.lean` returns one match, in a comment string ("zero sorry obligations") rather than a proof, and `build_full.log` reports `EXIT=0`. The mechanised assume/guarantee composition theorem on the 17-operator DSL is a substantive artifact.
- The paper acknowledges and quantifies the un-instantiated/class-source value proposition honestly: it does not paper over the $0$-unconditional-RP rate on the $488$-block real corpus, and pairs the headline numbers to clearly-named regimes (free-symbolic-config vs.\ synthesised envelope).
- The $32/34$ vs.\ $25/34$ Pytea head-to-head on the fragment-fair subset is a properly matched-pair comparison with both McNemar exact $p$ and the alternative silent-skip convention recorded.

## Weaknesses
- \textbf{Axiom~\ref{ax:operator-agnostic-witness} silently inflates Theorem~\ref{thm:ag-sound}'s scope.} The mechanised composition theorem is advertised as covering $17$ operators including \texttt{matmul}, but \texttt{matmul} and \texttt{broadcast\_add} — arguably the two most semantically loaded ops — are discharged by an "operator-agnostic composition witness" whose first clause is literally "the rule-table shape function agrees with the runtime shape on every input multiset that satisfies the rule's precondition (this is the in-envelope agreement count of $1{,}000$/$1{,}000$ samples)". Using a $1000$-sample property test as an \emph{axiom} for matmul inside a theorem stated in Lean is a soundness-grade move, not a presentation issue. Either the lemmas \texttt{applyOpExt\_sound\_matmul}/\texttt{\_broadcast\_add} should be closed in Lean, or the theorem should be restated as covering $15$ operators with $2$ explicit conjectures, and the abstract's "$15$ per-operator lemmas and $2$ explicit operator-agnostic obligations" claim should not be folded into a sentence whose grammatical subject is a "mechanised … composition theorem on a $17$-operator DSL".
- \textbf{The model-extraction definition (\Cref{def:model-extraction}) is mathematically incomplete on the grad component.} The definition writes $\mathit{requires\_grad}=\mathfrak{m}(g_i)$ where $g_i$ is a refinement in the three-element lattice $\{\mathsf{has\_grad},\mathsf{no\_grad},\top\}$. There is no statement of which Z3 sort encodes $g_i$, how $\top$ is realised in a concrete heap (it is not Boolean), or why the negated obligation in case (ii) is jointly satisfiable with the per-tensor refinements when the witnessed obligation is grad-only. Lemma~\ref{lem:progress-to-witness} silently inherits the gap.
- \textbf{The "soundness theorem" of \Cref{thm:soundness} is conditional on three load-bearing TCB obligations enumerated in \Cref{rem:tcb-thm-ii}}, of which (a) — that the runtime PyTorch handler raises on $\neg\varphi_{\mathrm{op}}$ — is the entire content of refutation soundness for the $16$ pen-and-paper handlers, and is supported only by "the corresponding PyTorch handlers are documented to raise on the same precondition" plus the same $1000$-sample agreement count as in Axiom~\ref{ax:operator-agnostic-witness}. There is no closed argument that documentation $+$ sampling implies refutation soundness; the theorem statement should be honest that case~(ii) for $\mathrm{Cat}_{\mathrm{pen}}$ rests on PyTorch documentation rather than on a derivation.
- \textbf{Axiom~\ref{ax:fresh-witness} (fresh-witness refutation) is an axiom about the implementation, not the calculus.} \Cref{thm:monotonicity} explicitly conditions its second clause on this implementation invariant being satisfied by the shipped analyser, then validates the invariant by a $200$-replay property test. A theorem whose statement \emph{names the current binary} as a hypothesis is operating outside the standard "calculus + mechanisation" mode the paper otherwise claims.
- \textbf{The headline empirical claims rest on heavily curated corpora.} The $53/60$ RP rate is on a corpus mined by 20+ keyword queries and filtered down from $1{,}087$ hits via four exclusion rules, of which (iv) ("config-attribute bugs that reduce to constructor sentinel-resolution rather than to shape arithmetic", $\sim 113$ hits) is exactly the regime where TG returns $0/113$ unconditional RP. The pre-registered post-freeze unfiltered $N{=}15$ sample collapses to $5/15$ ($33.3\%$) and is not statistically separable from \texttt{FakeTensorMode}'s $2/15$ or Pytea's $3/15$ at $\alpha{=}0.05$ (BH-adjusted Fisher $p{=}1.00$). The bug-corpus headline is therefore conditional on excluding the regime in which the analyser returns nothing.
- \textbf{Pytea baseline is essentially abandoned.} The paper concedes the upstream Pytea repository "has zero commits after \texttt{cb02a8a} (2022-04-26)". A $94.1\%$ vs.\ $73.5\%$ comparison against a four-year-stale baseline is descriptive at best; it is presented as a load-bearing head-to-head with $p$-values. The contemporary baseline that does work — \texttt{torch.compile} — beats TG ($34/34$ vs.\ $32/34$) on the same fragment-fair subset, a fact buried in the "setting asymmetry" paragraph.
- \textbf{The relationship between Theorem~\ref{thm:soundness} and the classical Preservation/Progress pair is asserted, not derived.} The "Subject reduction and progress" paragraph (\S\ref{sec:calculus}) claims Theorem~\ref{thm:soundness} is "the abstract-interpretation specialisation of this classical pair to the verdict lattice", with case (ii) (witness extraction) added on. Case (ii) is not a specialisation of progress/preservation — it is an extra "completeness for refutation" claim that requires the model-extraction lemma plus the in-envelope semantics agreement of the runtime, neither of which is part of subject reduction. The framing oversells the proof structure.

## Questions
- For \Cref{def:model-extraction}: please give the explicit Z3 encoding of the grad refinement $g_i$, the resolution of $\mathfrak{m}(g_i)$ when the verdict is grad-only and $g_i=\top$, and a closed argument that $\sigma_{\mathfrak{m}}\models\Gamma$ holds when the negated obligation $\neg\varphi_{\mathrm{op}}$ is jointly satisfiable but only with $\mathfrak{m}(g_i)=\top$.
- Why are \texttt{matmul} and \texttt{broadcast\_add} not given closed Lean \texttt{applyOpExt\_sound\_*} lemmas? They are arithmetically simpler than \texttt{conv2d} and \texttt{einsum}, both of which are mechanised. Is there a specific obstacle, or simply unfinished work? If unfinished, the theorem should be restated as $15$-of-$17$.
- Restricting to the pre-registered, unfiltered post-freeze $N{=}15$ sample, what is the per-bug verdict pair against \texttt{torch.compile}, and what fraction of the $9/15$ silent-verifieds correspond to bugs whose forward body uses only handlers in $\mathrm{Cat}_{\mathrm{sound}}$? Without this we cannot distinguish "fragment too small" from "rules in-fragment but unsound on this case".
- For the $113$-bug "config-attribute" exclusion slice on which TG returns $0/113$ unconditional RP and $16/113$ silent-verifieds: are the $16$ silent-verifieds inside or outside $\mathrm{Cat}_{\mathrm{sound}}$? A silent verified that fires through audited handlers is a far more serious soundness-direction observation than one that fires through tested-only handlers.
- The Lean composition theorem is on a $17$-operator DSL whose operators do not include attention, normalisation, or any reduction with non-trivial broadcasting (only \texttt{sum\_reduce}/\texttt{mean\_reduce}). What is the obstacle to mechanising the assume/guarantee theorem on the operator set that actually fires on the headline bug-corpus catches (\texttt{batch\_norm}, \texttt{cross\_entropy}, \texttt{max\_pool2d})?
- On the $46/56$ "in-soundness footprint" decomposition of bug-corpus catches: please give the full table mapping each of the $56$ catches to (primary handler, sub-catalogue), so the $46$ figure is independently checkable rather than asserted.

## Scores
Soundness: 2
Presentation: 3
Contribution: 3
Confidence: 3
Overall: 5

## Borderline reasons
The single change that would move me from 5 to 6 is closing Axiom~\ref{ax:operator-agnostic-witness} for \texttt{matmul} and \texttt{broadcast\_add} in Lean — i.e.\ promoting Theorem~\ref{thm:ag-sound} from a "$15$+$2$-axiom" theorem to an unconditional theorem on its stated operator set — and tightening \Cref{def:model-extraction} to explicitly handle the three-valued grad lattice. With those changes, the mechanisation story would actually match the prose, and the 88.3% headline number would have a defensible soundness floor for the catches it cites.


Changes   +0 -0
Requests  7.5 Premium (2m 28s)
Tokens    ↑ 715.2k • ↓ 8.1k • 633.2k (cached)

### Active obligations
- [reviewer, w=1.00, added round 1, streak=0] \textbf{Axiom~\ref{ax:operator-agnostic-witness} silently inflates Theorem~\ref{thm:ag-sound}'s scope.} The mechanised composition theorem is advertised as covering $17$ operators including \texttt{matmul}, but \texttt{matmul} and \texttt{broadcast\_add} — arguably the two most semantically loaded ops — are discharged by an "operator-agnostic composition witness" whose first clause is literally "the rule-table shape function agrees with the runtime shape on every input multiset that satisfies the rule's precondition (this is the in-envelope agreement count of $1{,}000$/$1{,}000$ samples)". Using a $1000$-sample property test as an \emph{axiom} for matmul inside a theorem stated in Lean is a soundness-grade move, not a presentation issue. Either the lemmas \texttt{applyOpExt\_sound\_matmul}/\texttt{\_broadcast\_add} should be closed in Lean, or the theorem should be restated as covering $15$ operators with $2$ explicit conjectures, and the abstract's "$15$ per-operator lemmas and $2$ explicit operator-agnostic obligations" claim should not be folded into a sentence whose grammatical subject is a "mechanised … composition theorem on a $17$-operator DSL".
- [reviewer, w=1.00, added round 1, streak=0] \textbf{The model-extraction definition (\Cref{def:model-extraction}) is mathematically incomplete on the grad component.} The definition writes $\mathit{requires\_grad}=\mathfrak{m}(g_i)$ where $g_i$ is a refinement in the three-element lattice $\{\mathsf{has\_grad},\mathsf{no\_grad},\top\}$. There is no statement of which Z3 sort encodes $g_i$, how $\top$ is realised in a concrete heap (it is not Boolean), or why the negated obligation in case (ii) is jointly satisfiable with the per-tensor refinements when the witnessed obligation is grad-only. Lemma~\ref{lem:progress-to-witness} silently inherits the gap.
- [reviewer, w=1.00, added round 1, streak=0] \textbf{The "soundness theorem" of \Cref{thm:soundness} is conditional on three load-bearing TCB obligations enumerated in \Cref{rem:tcb-thm-ii}}, of which (a) — that the runtime PyTorch handler raises on $\neg\varphi_{\mathrm{op}}$ — is the entire content of refutation soundness for the $16$ pen-and-paper handlers, and is supported only by "the corresponding PyTorch handlers are documented to raise on the same precondition" plus the same $1000$-sample agreement count as in Axiom~\ref{ax:operator-agnostic-witness}. There is no closed argument that documentation $+$ sampling implies refutation soundness; the theorem statement should be honest that case~(ii) for $\mathrm{Cat}_{\mathrm{pen}}$ rests on PyTorch documentation rather than on a derivation.
- [reviewer, w=1.00, added round 1, streak=0] \textbf{Axiom~\ref{ax:fresh-witness} (fresh-witness refutation) is an axiom about the implementation, not the calculus.} \Cref{thm:monotonicity} explicitly conditions its second clause on this implementation invariant being satisfied by the shipped analyser, then validates the invariant by a $200$-replay property test. A theorem whose statement \emph{names the current binary} as a hypothesis is operating outside the standard "calculus + mechanisation" mode the paper otherwise claims.
- [reviewer, w=1.00, added round 1, streak=0] \textbf{The headline empirical claims rest on heavily curated corpora.} The $53/60$ RP rate is on a corpus mined by 20+ keyword queries and filtered down from $1{,}087$ hits via four exclusion rules, of which (iv) ("config-attribute bugs that reduce to constructor sentinel-resolution rather than to shape arithmetic", $\sim 113$ hits) is exactly the regime where TG returns $0/113$ unconditional RP. The pre-registered post-freeze unfiltered $N{=}15$ sample collapses to $5/15$ ($33.3\%$) and is not statistically separable from \texttt{FakeTensorMode}'s $2/15$ or Pytea's $3/15$ at $\alpha{=}0.05$ (BH-adjusted Fisher $p{=}1.00$). The bug-corpus headline is therefore conditional on excluding the regime in which the analyser returns nothing.
- [reviewer, w=1.00, added round 1, streak=0] \textbf{Pytea baseline is essentially abandoned.} The paper concedes the upstream Pytea repository "has zero commits after \texttt{cb02a8a} (2022-04-26)". A $94.1\%$ vs.\ $73.5\%$ comparison against a four-year-stale baseline is descriptive at best; it is presented as a load-bearing head-to-head with $p$-values. The contemporary baseline that does work — \texttt{torch.compile} — beats TG ($34/34$ vs.\ $32/34$) on the same fragment-fair subset, a fact buried in the "setting asymmetry" paragraph.
- [reviewer, w=1.00, added round 1, streak=0] \textbf{The relationship between Theorem~\ref{thm:soundness} and the classical Preservation/Progress pair is asserted, not derived.} The "Subject reduction and progress" paragraph (\S\ref{sec:calculus}) claims Theorem~\ref{thm:soundness} is "the abstract-interpretation specialisation of this classical pair to the verdict lattice", with case (ii) (witness extraction) added on. Case (ii) is not a specialisation of progress/preservation — it is an extra "completeness for refutation" claim that requires the model-extraction lemma plus the in-envelope semantics agreement of the runtime, neither of which is part of subject reduction. The framing oversells the proof structure.
- [reviewer, w=1.00, added round 1, streak=0] For \Cref{def:model-extraction}: please give the explicit Z3 encoding of the grad refinement $g_i$, the resolution of $\mathfrak{m}(g_i)$ when the verdict is grad-only and $g_i=\top$, and a closed argument that $\sigma_{\mathfrak{m}}\models\Gamma$ holds when the negated obligation $\neg\varphi_{\mathrm{op}}$ is jointly satisfiable but only with $\mathfrak{m}(g_i)=\top$.
- [reviewer, w=1.00, added round 1, streak=0] Why are \texttt{matmul} and \texttt{broadcast\_add} not given closed Lean \texttt{applyOpExt\_sound\_*} lemmas? They are arithmetically simpler than \texttt{conv2d} and \texttt{einsum}, both of which are mechanised. Is there a specific obstacle, or simply unfinished work? If unfinished, the theorem should be restated as $15$-of-$17$.
- [reviewer, w=1.00, added round 1, streak=0] Restricting to the pre-registered, unfiltered post-freeze $N{=}15$ sample, what is the per-bug verdict pair against \texttt{torch.compile}, and what fraction of the $9/15$ silent-verifieds correspond to bugs whose forward body uses only handlers in $\mathrm{Cat}_{\mathrm{sound}}$? Without this we cannot distinguish "fragment too small" from "rules in-fragment but unsound on this case".

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
