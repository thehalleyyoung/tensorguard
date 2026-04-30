● Read round15_reviewer.md
  │ .comet_neurips/round15_reviewer.md
  └ 119 lines read

● List directory .
  └ 55 files found

● Search (glob)
  │ "reproducibility/*.md"
  └ 96 files found

● Search (glob)
  │ "docs/paper/sections_v5/*.tex"
  └ 29 files found

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ 1 line read

● Read mutation_kill_rate_corpora.md
  │ reproducibility/mutation_kill_rate_corpora.md
  └ 51 lines read

● Read E_dynamo.tex
  │ docs/paper/sections_v5/E_dynamo.tex
  └ 109 lines read

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ 135 lines read

● Read surviving_mutants_handler_classification.md
  │ reproducibility/surviving_mutants_handler_classification.md
  └ 89 lines read

● Read contemporary_baseline_34.md
  │ reproducibility/contemporary_baseline_34.md
  └ 77 lines read

● Read hf_extra_families_round11.md
  │ reproducibility/hf_extra_families_round11.md
  └ 83 lines read

● Read per_block_user_visible_rp.md
  │ reproducibility/per_block_user_visible_rp.md
  └ 58 lines read

● Read block_corpus_488_reconciliation.md
  │ reproducibility/block_corpus_488_reconciliation.md
  └ 77 lines read

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ 186 lines read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:120 (120 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L120:300 (181 lines read)

● Read paper_artifact_reconciliation.md
  │ reproducibility/paper_artifact_reconciliation.md
  └ 60 lines read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L300:500 (201 lines read)

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ L1:80 (80 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L500:650 (151 lines read)

● Read cross_family_natural_bugs.md
  │ reproducibility/cross_family_natural_bugs.md
  └ 65 lines read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L650:800 (151 lines read)

● Read pytea_modern_mcnemar.md
  │ reproducibility/pytea_modern_mcnemar.md
  └ 62 lines read

## Summary
TensorGuard (TG) is a static shape-and-gradient verifier for PyTorch `nn.Module` classes that operates from class source alone (no instantiation, no concrete inputs). It assigns a refinement type `Tensor{s,g|φ}` to every value in `forward`, discharges shape obligations via Z3, and emits one of five verdicts (Verified, Refuted-Proof, Contract-Violation, Library-Warn, Abstain). Contributions include a formal refinement calculus with Preservation/Progress, a Lean-mechanized assume/guarantee composition theorem on a 17-operator DSL, an autograd-aware backward verifier, and empirical evaluation on a 488-block real-source corpus plus a 60-bug historical corpus. The key practical findings are a ≥481/488 inapplicability gap for execution-based baselines, 53/60 RP on the bug corpus, 7/7 RP on naturally-occurring cross-family upstream bugs, and a 5/15 catch rate on a pre-registered unfiltered post-freeze sample (not statistically separable from execution-based baselines at α=0.05).

## Prior weakness disposition
- [PARTIAL] The most important practical limitation remains: 0 unconditional RP on the free-symbolic 488-block corpus; 23/57 Verified assume-dependent -- The paper now provides per-block survival breakdown (34 survive/23 collapse) and a "two denominators" rerun (15/295 RP with input-shape contract), but the core 0-RP natural-distribution finding is unchanged.
- [RESOLVED] Cross-family decoder study RP counts driven by deliberately broken variants -- The paper now has a "naturally-occurring cross-family bugs" section with 7/7 RP on genuine upstream HF PRs and issues (Llama, Qwen2, Mistral, Phi-3), each with explicit PR/issue citation.
- [PARTIAL] Theorem 5 (Dynamo) uses mostly `trusted` contracts; n=14 modules audit -- Table now covers 17 modules (expanded by 3), but 16/17 still use "trusted" surrogate contracts and the Introduction still states "audited on a 14-module corpus," conflicting with the table.
- [UNRESOLVED] Robustness: 7/50 mutants killed at union of three corpora (14%) -- The surviving-mutant handler classification is now provided (18/43 structurally "false-RP capable"), but the kill rate itself is unchanged.
- [RESOLVED] Released artifacts not audit-friendly: two inconsistent headline triples (50/213/225 vs 57/206/225) -- The paper now has an explicit bookkeeping paragraph distinguishing HCO=True (57/206/225) vs HCO=False (50/213/225) with both counts cited; reconciliation artifact provided.
- [RESOLVED] `torch.compile`+FakeTensor catches 34/34 vs TG's 32/34 on the modern subset -- The paper now explicitly reports this in the contemporary-baseline paragraph, repositions TG's contribution as the class-source-only regime where torch.compile is inapplicable, and adds both tools to the comparison table.

## Strengths
- Unusually disciplined calibration: the five-way verdict taxonomy, the prominent 0-RP acknowledgment on the natural-distribution corpus, the complete disclosure of 10/128 single-default-omitted CVs, and the honest reporting of the 2/8 = 25% worst-case grad false-verified rate demonstrate reporting standards rarely seen at this detail level.
- The 7/7 RP result on naturally-occurring cross-family bugs from real upstream HF PRs (not injected variants) is a concrete upgrade to the practical bug-finding story, with individually verifiable PR/issue citations.
- Lean mechanization is solid: 11/11 previously-axiomatic soundness lemmas closed sorry-free, 15/17 operators with per-operator soundness witnesses, clean `lake build`; the AST-extractor oracle cross-validation (140/140 zero over-extractions) audits the unverified TCB link on the CV soundness chain.
- The pre-registered unfiltered post-freeze evaluation (N=15 with power calculations for a second wave) sets a commendable benchmark for honest post-freeze calibration.

## Weaknesses
1. **Internal module-count inconsistency (§1 vs §5)**: The Introduction (C4) describes the Dynamo lemma as "empirically audited on a 14-module corpus," but §5 opens "17 real modules" and Table~1 of that section contains 17 rows. As written, a reader of the introduction has a different corpus size in mind than a reader of §5; this is a factual inconsistency in a claim the intro presents as a headline calibration for C4.

2. **Dynamo section: 16/17 modules use surrogate "trusted" contracts, not TG-issued contracts**: The claimed Dynamo-guard inclusion is tested by verifying that documented-signature (surrogate) contracts predict guard stability—not that TG's own emitted contracts do. The single TG-verified row is TinyMLP, a hand-designed micro-module that trivially admits the static-shape fragment. The paper acknowledges this scoping but does not report what fraction of TG-issued contracts on real torchvision/HF models would pass the same guard-stability test, leaving the practical scope of C4 unclear.

3. **Mutation kill rate 14% (7/50) is unchanged; the "structural upper bound" claim for surviving mutants is not validated at the case level**: Among 43 surviving mutants, 18 are in the "other" or "z3-dispatch" families and classified as "structurally false-RP capable." The paper claims this "overstates the realised exposure" because these functions are exercised on the clean baseline without producing spurious RP, but no individual mutant is traced to show the specific mutated branch does not reach the RP-emitting decision. The structural argument is sound in principle but remains unverified at the level of the 18 specific sites.

4. **Pytea comparison uses two incompatible counts (22/34 and 25/34) in different artifacts without a single canonical note in the main text**: `contemporary_baseline_34.md` reports Pytea 22/34; `pytea_modern_mcnemar.md` explains 22 is the silent-skip-reclassified internal figure and 25 is the conservative headline. This discrepancy is buried in reproducibility files; the main eval table shows 25/34 with no footnote distinguishing the two. A reader who checks both artifacts encounters an unexplained 3-catch discrepancy in the tool the paper is being compared against.

5. **The post-freeze unfiltered result (5/15 TG vs 2/15 FakeTensorMode vs 3/15 Pytea) does not survive BH correction (all adjusted p = 1.00)**: The paper reports this correctly in §4 but the Conclusion (§6) restates the findings as point estimates without a matching hedging sentence on the post-freeze comparison. The conclusion "TG reorganises sound static verification … places its static refinements in necessary-direction correspondence with TorchDynamo guards" is accurate, but the practical bug-finding advantage over execution-based baselines on the unfiltered surface is stated as a finding when it is only a trend.

6. **W1 residual (natural-distribution utility)**: On the 488-block free-symbolic-config corpus, 0/488 unconditional RP verdicts are issued and 0/488 user-visible refutations occur. The 15/295 RP rate on the "two denominators" rerun requires a user-supplied input-shape contract that 481/488 real blocks lack by construction. The practical bug-finding utility of TG on code-in-the-wild, as opposed to its verification utility (53/60 on the curated bug corpus), remains unevidenced on the natural-distribution setting.

## Questions
1. C4 in the Introduction reads "empirically audited on a 14-module corpus" but §5 says "17 real modules" and Table~1 has 17 rows. Which count is authoritative, and should the introduction be updated?

2. For the 18 "structurally false-RP capable" surviving mutants, can you exhibit, for even a representative subset of 3–4 spanning different families ("other" vs "z3-dispatch"), the specific mutated branch and the corpus input(s) on which that branch is exercised, confirming no false RP is emitted? A per-family characterization rather than a new experiment would close the structural-vs-empirical gap.

3. The 7 naturally-occurring cross-family bugs (§4, "Naturally-occurring cross-family bugs") are all from HuggingFace transformers decoder families (Llama, Qwen2, Mistral, Phi-3). Are any naturally-occurring bugs from a genuinely different codebase ecosystem (e.g., timm vision models, PyTorch Geometric, diffusers non-UNet) included in any evaluation set, or are all out-of-training-corpus bugs either injected variants or from the HF decoder family?

4. The three post-freeze silent misses (rb_pf_002, rb_pf_005, rb_pf_006) are attributed to the "constructor-bound integer-attribute envelope" class. For each: is the missing rule a named per-rule strengthening within the current LIA∪Div∪BMul fragment (analogous to the 12-entry LW→RP candidate table), or does it require a fragment-level extension? A one-row extension to the LW→RP table would let readers assess whether the 3/6 post-freeze miss rate is recoverable without a theoretical advance.

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons
Reconciling the 14 vs 17 Dynamo module count between the Introduction and §5 is a one-line textual fix that removes the most immediately checkable factual discrepancy in the paper; adding a footnote to the headline Pytea comparison noting both the conservative (25/34) and the silent-skip-reclassified (22/34) numbers would eliminate the unexplained discrepancy between the main text and the `contemporary_baseline_34.md` artifact, and together these two editorial corrections would meaningfully increase reviewer confidence in the broader numerical claims.


Changes   +0 -0
Requests  1 Premium (4m 10s)
Tokens    ↑ 443.9k • ↓ 12.2k • 362.1k (cached)
