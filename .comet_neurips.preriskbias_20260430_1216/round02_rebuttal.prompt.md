# Role: paper authors writing a brief rebuttal

A NeurIPS reviewer just posted the review below on your paper. Before
you start any code or paper changes this round, you have ONE chance to
push back on weaknesses you believe are misweighted, factually wrong,
or already-resolved-in-the-current-repo. The next round's reviewer
WILL read this rebuttal and must either accept it (drop the weakness)
or sharpen it (restate with a concrete counter-example).

## The review you are rebutting
## Summary
This paper presents **TensorGuard**, a static verifier for PyTorch `nn.Module` **class source** that reasons about tensor shapes and gradient-flow properties without instantiating the module or executing example inputs. The core technical story is a refinement-typed calculus with Z3-backed obligations, an assume/guarantee discipline at module boundaries, and a five-way verdict taxonomy that explicitly distinguishes unconditional refutations from contract-dependent ones and abstentions. On the empirical side, the paper claims strong results on a curated 60-bug historical corpus, explicit calibration on a 488-block real-source corpus, additional evaluations on real public-repo bug re-extracts and post-freeze holdouts, and several baseline comparisons. It also reports a Lean audit of a subset of the operator-rule table and a separately scoped, SHA-pinned empirical correspondence between TensorGuard’s refinement variables and TorchDynamo’s guard reads. A recurring theme of the current draft is claim calibration: the paper now tries to separate what is mechanized, what is pen-and-paper, what is tested-only, and what is exploratory.

## Prior weakness disposition
- [RESOLVED] **Theorem 1 over-promises relative to its own sketch.** -- rebuttal accepted: the theorem now quantifies over $\mathrm{Cat}_{\mathrm{sound}}=\mathrm{Cat}_{\mathrm{audit}}\cup\mathrm{Cat}_{\mathrm{pen}}$ (44 handlers), so the boundary is in the statement rather than only in nearby prose.
- [RESOLVED] **Theorem 2 has the same internal contradiction.** -- The current soundness theorem/proof explicitly reduce only to Lean-audited and pen-and-paper handlers, removing the earlier 28/79 mismatch.
- [RESOLVED] **Theorem 4 (monotonicity) cites a "rely/guarantee axiom of fresh refutation witnesses needed to make Theorem 4 hold"...** -- The fresh-witness axiom is now stated immediately before the monotonicity theorem, so the dependency is no longer deferred to an off-page unstated assumption.
- [RESOLVED] **The 16 "pen-and-paper" handlers occupy a non-trivial slice of the soundness story but their proofs are not in the main theorem...** -- Table 8 now separates Lean / pen-and-paper / tested-only scopes and the theorem proof points to closed appendix sketches, so the paper no longer blurs these rows together.
- [PARTIAL] **The AST extractor cross-validation does not retire the TCB concern it claims to retire.** -- rebuttal rejected: the claim is now better calibrated and the added 20/20 hand-labelled slice helps, but an author-built oracle plus author hand labels still do not independently retire extractor-TCB risk.
- [RESOLVED] **Theorem 5 (Dynamo correspondence) is reported as a theorem but proved by inspection against a single PyTorch release.** -- rebuttal accepted: this is now a SHA-pinned empirical proposition/exploratory audit rather than a release-agnostic theorem, and the end-to-end audit makes that status explicit.
- [PARTIAL] **The headline `0/488` unconditional RP under the user-visible regime substantially undercuts the bug-finding narrative.** -- The framing is now much cleaner, but the natural-source class-level number is still 0/488 user-visible RP, so the practical bug-finding story still rests mostly on curated or reduced corpora.
- [RESOLVED] **Constants and assumptions in the typing rules are under-specified.** -- The paper now spells out broadcast semantics, the single-`-1` requirement, and the `Q\neq 0` / multi-unknown rejection cases for `view`/`reshape`.

## Strengths
- The paper is unusually **well calibrated empirically**: it distinguishes RP/CV/LW/Abstain, fronts the `0/488` user-visible limitation, and generally avoids hiding inapplicability behind optimistic aggregate numbers.
- The baseline story is materially stronger than before: Pytea is compared on a fragment-fair modern subset with paired statistics, and execution-based baselines (`torch.compile`, `FakeTensorMode`, `torch.fx`+ShapeProp) are run rather than merely cited.
- The artifact is rich enough that many headline tables appear **reconstructable from released JSON/scripts**, and the reproducibility statement is substantially better than average.
- The paper now does a much better job separating **mechanized**, **pen-and-paper**, **tested-only**, and **exploratory** claims, which is important for both soundness and empirical interpretability.

## Weaknesses
- The main practical limitation remains central: in the user-visible regime on the unreduced **488-block real-source corpus**, TensorGuard still reports **0/488 unconditional RP**, so the strongest bug-finding evidence comes from the curated 60-bug corpus, 10 upstream-faithful re-extracts, and reduced cross-family repros rather than from natural-distribution class source.
- The real-bug evidence is still **small-N**: the upstream-faithful table is `7/10` at `>=0.99` plus `1/10` at `0.80`, and the unfiltered post-freeze result is `5/15`, which the paper itself says is **not statistically separable** from FakeTensorMode or Pytea after correction.
- The ablation story is weak on natural workloads: Section 4.4 states that the five-knob ablation on the `488+60` corpora is a **flat line**, and the discriminative evidence comes only from a hand-designed **25-case stress benchmark**.
- The Dynamo section is better framed now, but much of the evidence is still **signature-trusted or audit-by-inspection** rather than end-to-end TG-generated contracts, and the larger falsifier audits mostly show absence of SHAPE/DTYPE/RANK falsifiers rather than strong practical usefulness.
- The released artifact still has at least one **stale internal inconsistency**: `experiments_v5/v8/lean_sorry_elim_report.json` reports one remaining `sorry`, while the live Lean sources/build log and the paper say the tree is sorry-free; this weakens confidence that every released auxiliary artifact is canonical.

## Questions
- For the **7 naturally occurring cross-family bugs**, what is TensorGuard’s catch rate on the **original upstream class source** before reducing each case to a self-contained minimal module?
- Can the authors provide one compact table aligning the **10 upstream-faithful**, **15 post-freeze**, **15 unfiltered**, and **7 cross-family natural** bug sets under the **same confidence threshold and same baselines**?
- Of the **12 named LW→RP candidates** on the 488-block corpus, how many become actionable RP in the **no-synthesised-assume user-visible regime**, rather than only in the input-shape-contract rerun?
- In the Dynamo section, can the paper surface in one place which rows are **TG-verified end-to-end** versus **signature-trusted**, and what the **timeout / warm-up-failure denominator** is in the larger population audits?

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons
What would move me from a 6 to a 7 is a **larger natural-distribution, source-level bug-finding result**: not more curated stress cases, but a materially bigger set of unreduced upstream class-source bugs showing nontrivial unconditional catches and a clear comparison to the strongest applicable baseline on the same denominator.

Round: 2


Changes   +0 -0
Requests  1 Premium (3m 37s)
Tokens    ↑ 635.8k • ↓ 11.0k • 551.2k (cached) • 6.3k (reasoning)

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
