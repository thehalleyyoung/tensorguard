# TensorGuard v5 — NeurIPS rewrite integration notes

This document describes how `neurips_v5.tex` is wired together,
which content comes from which Track, and the cross-section
concerns the Track authors should be aware of.

## Outline and page budget (≤9 pp body)

| § | Title | Source | Budget |
|---|-------|--------|--------|
| 1 | Introduction | `sections_v5/intro.tex` (this scaffold) | ~1.0 pp |
| 2 | Background | `sections_v5/background.tex` (TBD) | ≤0.7 pp |
| 3 | Refinement-typed calculus | `sections_v5/A.tex` (Track A) | ≤2.5 pp |
| 4 | Assume/guarantee composition | `sections_v5/B.tex` (Track B) | ≤1.5 pp (mergeable into §3 if tight) |
| 5 | Implementation: extended fragment + backward verifier | `sections_v5/C-summary.tex` (Track C) + `sections_v5/D-summary.tex` (Track D) | ≤1.5 pp combined |
| 6 | Empirical evaluation | `sections_v5/F.tex` (benchmark) + `sections_v5/E.tex` (Dynamo correspondence) + `sections_v5/G.tex` (Lean parity) | ≤2.0 pp combined |
| 7 | Related work | `sections_v5/H.tex` (Track H) | ≤0.6 pp |
| 8 | Limitations and conclusion | `sections_v5/limitations.tex` + `sections_v5/conclusion.tex` (this scaffold) | ≤0.4 pp combined |

Estimated body total at upper bounds: 1.0 + 0.7 + 2.5 + 1.5 + 1.5
+ 2.0 + 0.6 + 0.4 = **10.2 pp**. To hit the 9 pp NeurIPS limit
we plan to merge §4 into §3 (Track B's composition rule fits
naturally as a final subsection of the calculus), saving ~0.7 pp,
and trim §5 by 0.3 pp via tighter prose. Target post-merge:
~9.0 pp.

## Per-section content map

### §1 Introduction (`intro.tex`, in this PR)
- Frames TensorGuard v5 as **refinement-typed shape & grad-flag
  verifier** with **assume/guarantee module composition**,
  **autograd-aware backward verification**, and a **Dynamo-guard
  correspondence**.
- Lists exactly **4 numbered contributions**:
  1. Refinement-typed calculus unifying shape + grad-flag.
  2. Assume/guarantee composition (`A ⇒ G` per `nn.Module`
     class, with sub-classing rule).
  3. Backward verifier catching the 3 canonical silent-zero-grad
     bugs (in-place leaf, `.detach()` on loss path,
     `torch.no_grad()` scope).
  4. Dynamo-guard correspondence + ≥150-block benchmark + 20-op
     Lean parity.
- Forward-references `\Cref{sec:calculus}`,
  `\Cref{sec:assume-guarantee}`, `\Cref{sec:impl-backward}`,
  `\Cref{thm:dynamo-corr}`, `\Cref{sec:eval}`,
  `\Cref{app:repro}`.

### §2 Background (TBD; expected `sections_v5/background.tex`)
- Brief PyTorch + autograd + Dynamo + refinement types.
- Should **not** re-derive the calculus; that's §3.

### §3 Refinement-typed calculus (Track A → `sections_v5/A.tex`)
- Owns the type grammar
  `τ = Tensor{s : Shape, g : Bool | φ}`, the typing judgement,
  the operator signature schema, and the soundness theorem
  against a small-step semantics with explicit autograd tape.
- Cross-section concern: §1 promises that a single Z3 procedure
  discharges both shape and grad refinements; Track A must
  state this explicitly so §5 can reuse it.

### §4 Assume/guarantee composition (Track B → `sections_v5/B.tex`)
- Module contract `A ⇒ G`, local composition rule,
  contravariant-input / covariant-output subclassing rule.
- Cross-section concern: must use the same `τ` from Track A.
  If Tracks A and B disagree on type-grammar notation, fix
  during integration.

### §5 Implementation (Tracks C and D)
- C summary: extended fragment, AST extractor, factory inlining,
  abstention triggers.
- D summary: backward verifier (abstract tape, severed-tape
  detection, witness format).
- Cross-section concern: §1 contribution (3) names the **3
  canonical silent-zero-grad bugs**; Track D must match exactly:
  *(a) in-place op on a leaf, (b) `.detach()` in the loss path,
  (c) `torch.no_grad()` scoping error.* If Track D's taxonomy
  differs, edit §1.

### §6 Empirical evaluation (Tracks F, E, G)
- F: ≥150-block benchmark (torchvision + HuggingFace + timm +
  bug injection); per-source in-fragment rate;
  Verified/Abstain/Rejected; head-to-head vs FX/FakeTensor/export.
- E: Dynamo-guard correspondence theorem + empirical agreement.
- G: 20 Lean-checked operator lemmas; build time; no `mathlib`.
- Cross-section concern: §1 promises **≥150 verification
  blocks** and **20 Lean operator lemmas**. If Tracks F or G
  finalise different counts, update the §1 contribution wording
  and the abstract.

### §7 Related work (Track H → `sections_v5/H.tex`)
- Refinement type systems for tensor programs; PyTorch's shape
  ecosystem; machine-checked tensor semantics. Distinguish from
  v4's related work, which only covered shape ecosystem peers.

### §8 Limitations and conclusion (this PR)
- Limitations: analyzed Python subset, abstention on truly
  dynamic control flow, backward verifier scope (not numerical
  correctness, not higher-order), Lean covers operator rules
  only, Dynamo correspondence is one-way at the boundary.
- Conclusion: one paragraph; no future-work puff.

## Cross-section concerns to flag during integration

1. **Notation unification.** Tracks A, B, D, E, G all reference
   the type grammar `Tensor{s,g | φ}`. The integrator must check
   that every Track uses the same metavariables (`s` for shape,
   `g` for grad flag, `φ` for the Z3 refinement). If Track A
   ships with different letters, normalise across the bundle.
2. **Theorem numbering.** §1 forward-references
   `\Cref{thm:dynamo-corr}`. Track E must label its main
   correspondence theorem `thm:dynamo-corr`. Similarly Track A
   should label its main soundness theorem `thm:soundness` (a
   v4-compatible label) so §5 and §8 references resolve.
3. **Operator-count consistency.** Abstract claims "$20$"
   Lean-checked operator lemmas; §1 contribution (4) repeats it.
   Track G is the source of truth — if the count changes, edit
   the abstract and §1 in lockstep.
4. **Benchmark-size consistency.** Abstract and §1 say
   "≥150 verification blocks". Track F is the source of truth.
   Edit abstract and §1 in lockstep if the actual count differs.
5. **Page budget.** §3+§4 are the largest risk; if Track A runs
   long, merge Track B's composition rule into §3 as planned
   above. If still over, demote one of the §6 sub-results to
   the appendix.
6. **Reproducibility.** Per the brief, no prominent in-body
   Reproducibility section. The skeleton points at
   `\Cref{app:repro}` from §1 only; do not re-introduce a body
   §Reproducibility.
7. **Order.** The skeleton enforces paper → references →
   appendix → checklist. Do not move the checklist; the NeurIPS
   style requires it last.
8. **Line numbers.** `\renewcommand{\sfdefault}{ptm}` follows
   `\usepackage{neurips_2026}` and precedes `\usepackage{lineno}
   \linenumbers`, so submission-mode line numbers render in the
   default PT-Sans-replacement font. Do not reorder.
9. **Conditional inputs.** Every Track input is wrapped in
   `\IfFileExists{...}{...}{<scaffold paragraph>}` so the
   document compiles at any point in the integration window.
   When a Track lands, its scaffold paragraph disappears
   automatically; nothing else needs to change.

## Files in this PR

- `docs/paper/neurips_v5.tex` — top-level skeleton.
- `docs/paper/sections_v5/intro.tex` — §1.
- `docs/paper/sections_v5/limitations.tex` — §8 limitations.
- `docs/paper/sections_v5/conclusion.tex` — §8 conclusion.
- `docs/paper/sections_v5/INTEGRATION_NOTES.md` — this file.

Not yet compiled; awaiting Tracks A–H.
