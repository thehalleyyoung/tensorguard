● Read round02_reviewer.md
  │ .comet_neurips/round02_reviewer.md
  └ 322 lines read

I’m pulling the paper claims and the artifact-backed numbers together so the review tracks the current repo rather than the prior round.

● Read headline_results.json
  │ benchmarks/headline_results.json
  └ 28 lines read

● Read pyproject.toml
  └ 44 lines read

● Read README.md
  └ L1:220 (220 lines read)

● Search (grep)
  │ "Theorem 1|Theorem 2|Theorem 4|Theorem 5|0/488|53/60|Table
  │ 8|handler-soundness|Boundary precision|Dynamo
  │ correspondence|audited|pen-and-paper|error bars|seed|reproducibility"
  │ (neurips.tex, neurips.txt, README.md, API.md)
  └ 7 lines found

● Read review_response.md
  └ L1:260 (260 lines read)

● Search (grep)
  │ "input\{|include\{|sections_v5" (neurips.tex)
  └ 9 lines found

● Search (grep)
  │ "Theorem 1|Theorem 2|Theorem 4|Theorem 5|Boundary precision|0/488|53/60|60
  │ bug|488|PyTea|seed|std|error bar|ablation|Table 8|handler-soundness|fresh
  │ refutation witnesses|pen-and-paper|torch\.2\.9\.1|Dynamo" (neurips.txt)
  └ 7 lines found

● Search (glob)
  │ "docs/paper/sections_v5/*.tex"
  └ 29 files found

I found the live section sources; next I’m checking the exact theorem wording, the evaluation tables, and whether the repo actually exposes enough to reconstruct the headline numbers.

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ L1:220 (136 lines read)

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ L1:260 (260 lines read)

● Read handler_soundness_table.tex
  │ docs/paper/sections_v5/handler_soundness_table.tex
  └ L1:240 (96 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:320 (320 lines read)

● Read appendix_v6.tex
  │ docs/paper/sections_v5/appendix_v6.tex
  └ L1:260 (260 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L320:640 (321 lines read)

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ L1:260 (192 lines read)

● Read README.md
  └ L220:340 (43 lines read)

● Search (glob)
  │ "experiments_v5/v8/**/*.json"
  └ 20 files found

● Search (glob)
  │ "reproducibility/*"
  └ 9 files found

● Read results.json
  │ experiments_v5/v8/feature_stress/results.json
  └ 1 line read

● Read pytea_modern_subset.json
  │ experiments_v5/v8/pytea_modern_subset.json
  └ 551 lines read

● Read user_visible_rp.json
  │ experiments_v5/v8/user_visible_rp.json
  └ 26 lines read

● Search (grep)
  │ "tab:ablation|ablation|feature stress|25-case|25 case|five
  │ knobs|low-confidence gating|device-consistency|gradient-flow"
  │ (docs/paper/sections_v5/eval_v6.tex, neurips.txt)
  └ 38 lines found

I’ve got enough of the empirical side. I’m now checking the two rebutted theory-adjacent items that still affect soundness scoring: the Theorem 5 framing and whether the “fresh witness” axiom is now actually stated where needed.

● Read dynamo_e2e_results.json
  │ experiments_v5/v8/dynamo_e2e/dynamo_e2e_results.json
  └ 260 lines read

● Read dynamo_e2e.md
  │ reproducibility/dynamo_e2e.md
  └ L1:240 (55 lines read)

● Read lean_sorry_elim_report.json
  │ experiments_v5/v8/lean_sorry_elim_report.json
  └ 89 lines read

✗ Search (grep)
  │ "thm:dynamo-corr|Dynamo|17-module|17 module|necessary
  │ direction|recompile|48/544|46/47|Theorem 5|fresh-witness|monotonicity|ax:fre
  │ sh-witness|11/11|sorry-free|32/57|30/128|66/185|57/185"
  │ (docs/paper/sections_v5/*.tex)
  └ rg: /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
    docs/paper/sections_v5/*.tex: IO error for operation on
    /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
    /paper/sections_v5/*.tex: No such file or directory (os error 2)

I hit one important inconsistency: one shipped JSON still reports a remaining Lean `sorry` even though the paper now says `11/11` are closed. I’m checking the live Lean sources rather than trusting the stale artifact.

● Read Extended.lean
  │ lean/TensorGuard/Extended.lean
  └ L200:260 (46 lines read)

● Read lake_build.log
  │ reproducibility/lake_build.log
  └ L1:120 (31 lines read)

● Search (grep)
  │ "^\s*sorry" in lean files (lean/TensorGuard)
  └ No matches found

● Search (grep)
  │ "theorem\[|Theorem 5|dynamo-corr|necessary
  │ direction|17-module|48/544|46/47|recompile rate|guard inclusion"
  │ (docs/paper/sections_v5)
  └ 31 lines found

● Read E_dynamo.tex
  │ docs/paper/sections_v5/E_dynamo.tex
  └ L1:180 (109 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1040:1210 (171 lines read)

● Read cegar_phase_deletion_tcb.py
  │ reproducibility/cegar_phase_deletion_tcb.py
  └ L1:220 (157 lines read)

● Read block_corpus_488_reconciliation.py
  │ reproducibility/block_corpus_488_reconciliation.py
  └ L1:220 (123 lines read)

One thing still looks potentially shaky: the paper’s Dynamo section mentions a 14-module extension, but the first artifact I opened only had the 5-module smaller-N audit. I’m checking whether the larger released audit actually exists before I treat that claim as supported.

● Read dynamo_theorem5_n200.md
  │ reproducibility/dynamo_theorem5_n200.md
  └ L1:260 (186 lines read)

● Search (grep)
  │ "14-module|14 module|55 successful modules|107 candidate modules|146
  │ candidate modules|19 recompile events|13 recompile events|3 sublayer
  │ modules|24/24" (reproducibility, experiments_v5/v8)
  └ 13 lines found

● Search (glob)
  │ "reproducibility/*dynamo*"
  └ 44 files found

● Search (glob)
  │ "experiments_v5/v8/**/*dynamo*.json"
  └ 3 files found

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
