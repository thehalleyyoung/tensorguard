● Read round03_reviewer.md
  │ .comet_neurips/round03_reviewer.md
  └ 133 lines read

● list repo and review folder (shell)
  │ ls -la
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/ |
  │ head -30 && echo "---" && ls /Users/halleyyoung/Documents/div/mathdivergence
  │ /halley-labs/tensorguard/.comet_neurips/ 2>/dev/null | head -30
  └ 45 lines...

● peek paper text (shell)
  │ wc -l /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguar
  │ d/neurips.txt && head -200 /Users/halleyyoung/Documents/div/mathdivergence/h
  │ alley-labs/tensorguard/neurips.txt
  └ 201 lines...

● section headers (shell)
  │ grep -n -E "^(##|Section|Theorem|Table)" /Users/halleyyoung/Documents/div/ma
  │ thdivergence/halley-labs/tensorguard/neurips.txt | head -80
  └ 14 lines...

● Read neurips.txt
  └ L800:1000 (201 lines read)

● Read neurips.txt
  └ L1240:1400 (161 lines read)

## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that statically reasons about symbolic shapes and a flat first-order grad-flag lattice (`{has_grad, no_grad, ⊤}`), discharging side conditions to Z3 and reporting a five-way verdict taxonomy. Empirics: 53/60 RP on a curated historical corpus; 32/34 vs. Pytea 22/34 on a fragment-fair modern subset (McNemar p=0.00195); 5/15 catches on the unfiltered pre-registered post-freeze N=15 PR sample (vs. FakeTensorMode 2/15, Pytea 3/15, Fisher non-separable; Bayesian BF₁₀=8.1 vs. FT, 3.6 vs. Pytea); 0 unconditional RP on the 488-block real-source corpus under the free-symbolic regime. A Lean 4 audit covers 28/79 shape-transfer handlers with 11/11 axiomatic lemmas closed sorry-free and 28,000/28,000 byte-mirror cases agreeing with torch 2.9.1. The Dynamo-guard correspondence (Thm. 5) is necessary-direction only and audited on ~31 modules total (17 original + 14 extended), with 19/19 recompiles classified `{SHAPE:19}` and zero out-of-catalogue guards.

## Prior weakness disposition
- [UNRESOLVED] the unfiltered pre-registered post-freeze evaluation is still N=15. The Bayesian supplement (BF₁₀=8.1 vs. FT, 3.6 vs. Pytea) does not exceed the conventional "st... -- Sample size remains N=15 with two-sided Fisher p=0.39 / 0.68; the Bayesian BFs sit in the "moderate" band, below the strong-evidence ≥10 threshold the prior round flagged.
- [PARTIAL] the 12-CV joint-realisability audit is a sample of 12/128 (~9.4%) selected as "12 randomly-sampled CV verdicts". The prior round asked for either the full 128-set or a uniformly... -- The 12-of-128 random sample with named-checkpoint pairings is the only joint-realisability evidence in the body; no full-128 ratio with CI and no scaled-up uniform subsample is reported.
- [PARTIAL] the Dynamo-falsification corpus is now ~31 modules (17 original + 14 extended), still well short of the ≥100 timm/HF blocks the prior round asked for, and 4 of the 14 exte... -- Total instantiated coverage remains ~31 modules; 4 of the 14 extended transformer blocks still rely on the forward-signature surrogate, so end-to-end ≥100-module evidence for Theorem 5 is still missing.
- [PARTIAL] the grad-flag silent-error audit ("0/16 `torch.utils.checkpoint`, 0/16 renamed-attribute parameter sharing") is a same-author pattern check on the same 17-module Theorem 5 fixture,... -- The audit fixture is unchanged; no held-out HF training-script false-verified-rate measurement appears, so the ≤12% prevalence claim still rests on the same-author sweep.
- [UNRESOLVED] The catalogue-coverage residual 12/78 "could-in-principle convert to RP" upper bound on the LW→RP gap is asserted in §4.1 but not exhibited at the per-block level... -- §4.1 still asserts the 12/78 residual without a per-block enumeration of which blocks would convert and under which missing rule.

## Strengths
- Calibration discipline remains unusually high: the 0 unconditional RP on the 488-block free-symbolic surface is reported as the headline; the 5/15 post-freeze catch is reported with explicit Wilson CI [15.2%, 58.3%] and explicit Fisher non-separability rather than as a separation claim; the off-axis fire on `rb_uf_010` is accounted as a false positive against ground truth and excluded from the headline catch count.
- The Lean audit is described accurately for what it certifies: 28/79 handlers, 11/11 axiomatic lemmas closed sorry-free, lake build sorry-free, and 28,000/28,000 byte-mirror against torch 2.9.1, with the analyser/AST extractor/backward verifier/Z3 dispatch explicitly held out as TCB. The `permList_compose_inrange` correction (replacing the originally false unconditional statement) and the boundary-mutator off-envelope check (no silent-through on ~2,400 samples across 10 rules) are non-trivial soundness work.
- C5 has been honestly narrowed: only the three discriminative knobs (device-consistency, gradient-flow, low-confidence gating) are claimed in the per-feature ablation; the unused CEGAR loop and always-satisfiable phase encoder are explicitly disclaimed as non-contributions, and the localisation tracer is relegated to engineering. This is the kind of negative framing reviewers usually have to drag out of authors.
- The fragment-fair Pytea head-to-head is methodologically tight: the 32/34 vs. 22/34 split on the modern subset is reproduced at verification time (TG restricted to the 2022 catalogue intersection at run time, AST-screen on each repro, forensics scan of `Bug.message`), the Pytea silent-skip correction is explicit, and the McNemar exact two-sided p=0.00195 with paired-bootstrap 95% CI [+14.7 pp, +44.1 pp] is a defensible statistical claim on this N.

## Weaknesses
- The post-freeze unfiltered evaluation is still N=15 (Section 4.1, Table 3). The BF₁₀=8.1 vs. FakeTensorMode and BF₁₀=3.6 vs. Pytea both sit in the "moderate" Jeffreys band, well below the conventional ≥10 strong-evidence threshold the authors themselves cite, and the frequentist Fisher tests (p=0.39 and p=0.68) do not separate. The empirical-superiority claim over execution-based baselines on the only unfiltered pre-registered surface is therefore still a point estimate, not a separation. Either extend the pre-registered query to N≥40 (which would, on the observed point estimates, push at least the TG-vs-FakeTensorMode comparison toward Fisher significance and BF₁₀≥10) and report the resulting numbers, or restate the headline as "point-above on N=15, not statistically separable" without leaning on the Bayesian supplement.
- The 488-block CV joint-realisability evidence (§4.1) is still the 12-of-128 random-sample audit (~9.4%) with named `*Config`-default instantiations and published checkpoints. The prior round explicitly asked for either all 128 or a substantially scaled-up uniform subsample with a confidence interval on the joint-realisability ratio. Without that, the "0 unconditional RP / 57 Verified" surface continues to depend on a `assume_M` whose *aggregate* realisability is estimated from <10% of the corpus. Please report the joint-realisability ratio over either all 128 CVs or a uniform random subsample of ≥60 with a Wilson/Clopper-Pearson CI on the realisable fraction.
- The Theorem 5 empirical audit is still ~31 modules total (17 original + 14 extended; §4.3 / "Extended end-to-end audit"). Of the 14 extended blocks, 4 transformer blocks are audited via the documented forward-signature surrogate, not a full instantiation, so the end-to-end count is closer to 9 CNN blocks + 1 ResNet50 layer + previously-audited cases. Theorem 5 is the central PL-side claim about Dynamo, and it is still being instantiated on roughly a third of the ≥100-block target the prior round named. Please run the falsifier on a further ≥70 importable timm/HF blocks (not the surrogate path) and report `{SHAPE, DTYPE, RANK, INT}` recompile counts and any out-of-catalogue guard hits.
- The grad-flag silent-error audit (§6) reports `0/16 torch.utils.checkpoint` and `0/16` renamed-attribute parameter sharing on the 16 importable Track-E modules — the same fixture used elsewhere in the paper. This is the only quantitative support for the headline "≤12% prevalence" caveat on the grad lattice's known unsoundness, and it is a same-author pattern check on a same-author fixture. Please run a held-out audit on a different population — e.g. the top-N HuggingFace `Trainer`/`accelerate` example training scripts — and report a false-verified-rate against runtime grad equality (not against a manually authored `# BUG` marker). A corpus around 30–50 training scripts would already discriminate ≤12% from, say, 25%.
- The "12/78 catalogue-coverage residual" bound on the LW→RP gap (§4.1) is asserted as an upper bound but not exhibited per-block. Without a list of which 12 of the 78 LW blocks would convert to RP under which specific missing rule, the bound is unfalsifiable from the paper alone — a future reviewer cannot tell whether the residual is realistic or whether several "in-principle convertible" entries are themselves blocked by the very catalogue gap that produced LW. Please add a small per-block table (block id → missing rule → predicted converted verdict) for the 12 cases.
- Theorem 1 (fragment-level soundness) and Theorems 10/11 (Preservation/Progress) are pen-and-paper, while Theorem 3 (compositional/assume-guarantee) is mechanised only on a 3-operator DSL via `lemma ag_composition`. The paper concedes this, but the TCB statement in §4.4 ("the analyser implementation, AST extractor, backward verifier, and Z3 dispatch remain in TCB") therefore covers the *entire* operational soundness story for the user-facing tool. A short explicit accounting of what survives if a TCB component is wrong (e.g. AST extractor mis-binds a starred view, backward verifier mis-classifies an in-place op) would calibrate what the 53/60 RP headline actually means about the artefact vs. about the calculus.

## Questions
- On the post-freeze unfiltered surface, what is the smallest N at which the observed point estimates (TG 1/3 vs. FT 2/15, Pytea 1/5) would yield BF₁₀≥10 and Fisher p<0.05 on at least one of the two pairwise comparisons? If extending the pre-registered query to that N is feasible, do so and report.
- For the 488-block corpus, what is the joint-realisability ratio of `assume_M` on either the full N=128 CV set or a uniform random subsample of N≥60, with a 95% CI? How many of the 12-of-128 named `*Config`-default instantiations remain joint-realisable when the published checkpoint's runtime input distribution (rather than the config defaults alone) is taken as the rely?
- Please tabulate the Theorem 5 falsification predicate over a further ≥70 end-to-end-instantiated importable blocks (not via forward-signature surrogate). What is the `{SHAPE, DTYPE, RANK, INT}` breakdown, and does any single block produce a SHAPE/DTYPE/RANK guard whose `guard_var` lies outside `catalogue(M)`?
- For the grad-flag lattice unsoundness on parameter-sharing-under-renamed-attribute, what is the false-verified-rate against runtime-observed `p.grad ≠ None` on a held-out corpus (e.g. ≥30 HF Trainer/accelerate example training scripts) as opposed to the same 16-module pattern sweep?
- Please give a per-block enumeration of the 12/78 LW→RP residual: for each of the 12 blocks, the missing operator handler whose addition would (in isolation) flip the verdict to unconditional RP, and the witnessing input shape.
- For the TCB components held out of the Lean audit (analyser, AST extractor, backward verifier, Z3 dispatch), what is the largest verdict-flip from a single deliberate fault-injection (e.g. AST mis-binding of `view(*new_shape)` star-expansion, backward verifier mis-classifying `Tensor.add_` as out-of-place) on the 60-bug and 488-block corpora?

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons
A single change would lift my score: extending either the unfiltered post-freeze sample to N≥40 (so that the BF₁₀≥10 / Fisher-significant separation claim against FakeTensorMode and Pytea becomes statistical rather than point-only), or the Theorem 5 end-to-end audit to ≥100 fully-instantiated importable blocks with the falsification predicate explicitly evaluated. Either would convert the most consequential PARTIAL/UNRESOLVED items into RESOLVED at the next round.

Round: 3


Changes   +0 -0
Requests  7.5 Premium (1m 52s)
Tokens    ↑ 273.4k • ↓ 5.5k • 243.6k (cached)
