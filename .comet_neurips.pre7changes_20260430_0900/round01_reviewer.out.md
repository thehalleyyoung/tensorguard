● Read round01_reviewer.md
  │ .comet_neurips/round01_reviewer.md
  └ 114 lines read

● List repo contents (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls && wc -l neurips.tex neurips.txt 2>/dev/null
  └ 17 lines...

● Read neurips.tex
  └ 90 lines read

● List sections and experiments (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls sections_v5/ 2>/dev/null && ls experiments_v5 2>/dev/null | head -30
  └ 1 line...

● Find sections_v5 directory (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && find . -path ./tensorguard.egg-info -prune -o -name "sections_v5" -print
  │ 2>/dev/null; ls experiments_v5
  └ 19 lines...

● Read neurips.txt
  └ L1:200 (199 lines read)

● Read neurips.txt
  └ L200:600 (401 lines read)

● Read neurips.txt
  └ L600:1100 (501 lines read)

● Read neurips.txt
  └ L1100:1600 (500 lines read)

● Read neurips.txt
  └ L1600:2100 (469 lines read)

● Inspect repo structure for claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/ 2>/dev/null && grep -r "sorry" lean/ 2>/dev/null | wc -l; ls
  │ reproducibility/ 2>/dev/null | head; ls real_benchmarks 2>/dev/null | head
  └ 19 lines...

● Check for sorry in lean (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -rn "sorry" lean/TensorGuard/ 2>/dev/null | head -30
  └ 8 lines...

● Check actual sorry occurrences and baseline data (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -rEn "^\s*(sorry|admit)\b" lean/ 2>/dev/null | head; ls
  │ experiments_v5/v8 2>/dev/null | head; cat
  │ experiments_v5/v5_baseline_comparison.json 2>/dev/null | head -50
  └ 61 lines...

## Summary
TensorGuard is a static, no-execution refinement-type checker for PyTorch `nn.Module` classes. From class source alone (no instantiation, no tracing) it computes refinement types `Tensor{s, g | φ}` over symbolic shape and a static gradient-required flag, dispatching shape obligations to Z3, and treats `nn.Module` boundaries as assume/guarantee contracts with contravariant/covariant subclassing. It introduces a five-way verdict taxonomy (V/RP/CV/LW/A) so that "verified" and "refuted" are only claimed under stated soundness conditions. Empirically the system reports 53/60 on a curated historical bug corpus, a 32/34 vs. 22/34 head-to-head against Pytea on a fragment-fair subset, 5/15 catches versus 2/15 (FakeTensorMode) and 3/15 (Pytea) on a pre-registered post-freeze unfiltered PR sample, and 0 unconditional RP on a 488-block real-source corpus (reported as a fragment-coverage measurement). 28 of 79 shape-transfer rules and an operator-agnostic composition lemma over a 13-operator DSL are mechanized in Lean 4 sorry-free; the analyser, AST extractor, backward verifier, and Z3 dispatch remain in the trusted computing base.

## Prior weakness disposition
(none — first round)

## Strengths
- Calibrated, honest reporting: a five-way verdict taxonomy (V/RP/CV/LW/A) with a precise theorem (Thm. 2) about which verdicts carry soundness claims, and explicit acknowledgement that the 488-block real-source corpus produces 0 unconditional RP under the user-visible free-symbolic regime. Selection effects (1087→60 keyword filter, exclusion rules iii–iv) are quantified and applied symmetrically to held-out corpora.
- Real, partial Lean mechanization: 28 shape-transfer rules and an `ag_composition_ext` theorem over a 13-operator DSL are sorry-free under `lake build` (verified in `lean/TensorGuard/`), with 28,000/28,000 byte-mirror agreement against torch 2.9.1 in-envelope plus a boundary check for ~2,400 off-envelope samples on 10 rules. The Lean operator registry is exported as JSON so the Python analyser cannot silently reference an undeclared op.
- Pre-registered post-freeze evaluation (catalogue freeze 2026-04-07; query frozen 2026-04-08) on `N=15` unfiltered merged PRs is methodologically uncommon for ML-tooling papers and provides genuine evidence against retro-fitting handlers.
- The `nn.Module` assume/guarantee discipline with the contravariant/covariant subclassing rule (Sec. 2.2) is a clean conceptual contribution that prior comparators (Pytea, FakeTensorMode, torch.export) do not articulate.
- Multi-pronged TCB stress evidence: four hand-picked single-fault injections plus a 50-mutant AST-mutation sweep across three corpora, with both exposure ceilings and measured RP→V flip counts reported per fault.

## Weaknesses
- **Headline real-world bug-finding result is weak.** On the only sample drawn without selection for fragment fit (Table 3, `N=15`), TG catches 5/15 vs. FakeTensorMode 2/15 and Pytea 3/15, neither pairwise gap statistically separable at α=0.05 (Fisher exact p=0.39, p=0.68). The 88.3% (53/60) and 94.1% (32/34) headlines rely on corpora the paper itself documents as filtered to operators the TG/Pytea fragments handle (exclusions iii+iv remove ~22% of hits, and the exclusion class is precisely where TG silently mis-verifies — Sec. 4.1, "0/113 unconditional RP on the config-attribute exclusion slice"). The contribution claim "53/60 RP" should be read against the 5/15 unfiltered number; the paper does so, but the abstract still leads with the curated figures.
- **The Lean mechanization claim does not extend to soundness of the deployed verifier.** Only 28/79 handlers are Lean-audited; the assume/guarantee composition theorem (Thm. 3) is mechanised on a 13-operator DSL, not on the 79-handler catalogue. The analyser implementation, AST extractor, backward verifier, and Z3 dispatch are explicitly outside the proof envelope. The end-to-end verdict an authoritative claim requires is therefore "Z3 + Python analyser + Lean-audited rule" — but the bug-firing path on most real catches traverses the unaudited components. The mutation-testing kill rate of 7/50 (14% best-of-three-corpora) suggests the analyser is not robust to single-line edits across most of its surface.
- **Theorem 5 (Dynamo-guard correspondence) is over-scoped relative to its evidence.** It is stated as a statement over "the supported fragment ∩ Dynamo's traceable subset" but on 16 of the 17 audit modules the contract is the documented `forward` *signature surrogate* rather than the full instantiated module; on the 55-module larger sweep, 0 SHAPE/DTYPE/RANK guards fired (only INT specialisations), so the falsification predicate is not actually exercised on that population. The headline "necessary direction holds" is supported by 13 SHAPE recompiles on 9 CNN blocks, which is a thin empirical base for a theorem about Dynamo's specialiser.
- **The 128 ContractViolation verdicts depend on a synthesised caller-rely envelope whose realisability is checked only against a single default `*Config()` instantiation.** 10/128 are unwitnessed even under that single instantiation (Sec. 4.1). Because CV is one of only two verdicts Theorem 2 covers, a CV count being inflated by liberal envelope synthesis is a soundness-relevant question, not a presentation one.
- **The first-order grad-flag lattice is admitted to be silently incorrect on parameter-sharing-under-renamed-attribute, with the prevalence bounded "≤ 12% of training scripts."** The paper later reports 0/2908 renamed-attribute hits in a separate AST-grep sweep and 1/42 in a held-out HF examples sweep, which makes the ≤12% ceiling appear conservative — but the lattice still produces *silent* (not Abstain) wrong results in this regime, and the population in which the silent-error regime is most consequential (full training pipelines using `torch.utils.checkpoint` plus tied weights) is not the one in which the 0/8 runtime false-verified rate is measured.
- **Two of the three "discriminative" features in the per-feature stress benchmark (Table 5) are admitted to be no-ops on real corpora.** CEGAR and phase-check are zero-delta on both 488-block and 60-bug corpora and on the 10-bug real-public corpus; only device-consistency, gradient-flow, and low-confidence gating discriminate, and only on the synthetic 25-case stress set. The flat real-corpus ablation undermines the claim that the engineering surface (CEGAR loop, phase encoder) contributes anything to the empirical headline.
- **Presentation.** The paper packs caveats into running prose to such a density that the actual claims become hard to extract (e.g. the LW→RP-candidate paragraph spans ~40 lines with one sentence of structural argument and the rest as parenthetical scope qualifications). Tables 1 and 4 are difficult to parse because counts are split across many sub-columns and footnotes.
- **The "32/34 vs. 22/34" McNemar result reports `b=10, c=0`** (Pytea-refutes is a strict subset of TG-refutes). This is presented confidently, but the paper does not show the per-bug agreement table that would let a reader verify the strict-subset claim independently of the protocol scripts.

## Questions
- Could you report the raw per-bug agreement matrix for the 34-bug fragment-fair head-to-head (Table 1 / Sec. 4.1) so that the strict-subset claim `c=0` is checkable without running the harness?
- For the 128 CV verdicts, can you give a multi-config realisability check — i.e. evaluate each `assume_M` against, say, 5 distinct published checkpoints' configs per HF backbone, and report the per-row witnessed rate? The single-default-config 118/128 number is the load-bearing soundness witness for CV.
- What is the recompile-classification breakdown on the 55-module Dynamo audit if the 240s wall-clock kill is raised to 1200s? The current 0/55 SHAPE-recompile result on the larger population could be a censoring artifact rather than a property of Dynamo's specialiser.
- What is the per-module false-positive rate of the `tensor.utils.checkpoint`/`gradient_checkpointing_enable` Abstain detector on a held-out positive set strictly larger than the 6-module hand-built and 8-module HF-head harness? A 0/8 runtime false-verified result is consistent with both "the detector is sound" and "the detector over-Abstains."
- Could you give a single number for the unfiltered post-freeze RP rate stratified by bug class (distributed/dtype/autograd-sharing/data-dependent/in-fragment), so the 5/15 headline can be read against the in-fragment denominator rather than against `N=15`?
- The paper claims the analyser implementation, AST extractor, and Z3 dispatch are in the TCB but bounds the worst-case impact via four hand-picked faults plus 50 random mutations. What is the kill rate on a stratified mutation sweep that targets the AST extractor and Z3-dispatch modules specifically (rather than the analyser core), and how does the upper-bound exposure on RP change?

## Scores
Soundness: 3
Presentation: 2
Contribution: 2
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would push this from 5 to 6 is replacing the curated `N=60` and `N=34` headlines with a substantially larger unfiltered post-freeze evaluation (e.g. `N≥75`) in which TG's catch rate over FakeTensorMode and Pytea is statistically separable at α=0.05 — the present `N=15` sample is honest but too small to carry the contribution, and the curated corpora have a documented selection alignment with the TG fragment that the unfiltered surface does not.


Changes   +0 -0
Requests  7.5 Premium (2m 48s)
Tokens    ↑ 701.5k • ↓ 6.9k • 643.4k (cached)
