● Read round01_reviewer.md
  │ .comet_neurips/round01_reviewer.md
  └ 113 lines read

● List dir and sizes (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls && wc -l neurips.tex neurips.txt README.md API.md 2>/dev/null
  └ 18 lines...

● Read neurips.tex
  └ 90 lines read

● Read neurips.txt
  └ L1:250 (249 lines read)

● Read neurips.txt
  └ L250:700 (451 lines read)

● Read neurips.txt
  └ L700:1200 (501 lines read)

● Read neurips.txt
  └ L1200:1700 (474 lines read)

● Explore structure (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/ 2>/dev/null && echo "---" && ls src/ | head -30 && echo "---" &&
  │ ls reproducibility/ 2>/dev/null | head -30 && echo "---" && wc -l src/*.py
  │ 2>/dev/null | tail -5
  └ 71 lines...

● Inspect Lean (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/TensorGuard/ 2>/dev/null && echo "---" && grep -l "sorry"
  │ lean/TensorGuard/*.lean 2>/dev/null && echo "---" && grep -c
  │ "^theorem\|^lemma" lean/TensorGuard/*.lean 2>/dev/null | head
  └ 12 lines...

● Check sorry usage (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "sorry" lean/TensorGuard/*.lean
  └ 7 lines...

● Check benchmarks dirs (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls benchmarks/ 2>/dev/null | head && echo "---" && ls real_benchmarks/
  │ 2>/dev/null | head && echo "---" && ls examples 2>/dev/null | head && echo
  │ "---" && ls tests 2>/dev/null | head -20
  └ 46 lines...

## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` classes that statically infers tensor shape and `requires_grad` refinements from class source. The system formalises a refinement-type calculus `Tensor[τ, φ_shape, φ_grad]` over an `nn.Module` fragment F_TG, with assume/guarantee composition at the class boundary, a backward-pass verifier for three silent-zero-grad bug classes, and a necessary-direction correspondence to TorchDynamo guards. A 28-rule subset of the 79-handler shape-transfer table is mechanised in Lean 4 with 11/11 soundness lemmas closed sorry-free. Empirically the paper reports 53/60 RP on a curated historical bug corpus, 32/34 vs Pytea 22/34 on a fragment-fair subset (McNemar p=0.00195), 5/15 catches on a pre-registered post-freeze unfiltered real-PR sample (vs FakeTensorMode 2/15 and Pytea 3/15, not statistically separable), and 0 unconditional RP on the 488-block real-source corpus under the free-symbolic-config regime.

## Prior weakness disposition
(none — first round)

## Strengths
- Unusually calibrated reporting: the paper does not hide that on the 488-block real corpus it issues 0 unconditional RP under the free-symbolic-config regime; verdicts are split RP/CV/LW/Abstain and only RP+CV are claimed sound (Sec 4.1, Table 1). This is rare in the ML-tooling literature.
- The Lean 4 audit is real and inspectable (`lean/TensorGuard/`, the tree builds sorry-free per `grep`-able comments and source); the byte-mirror cross-check against torch 2.9.1 on 28,000 in-fragment samples is a meaningful operator-rule sanity check.
- The fragment-fair head-to-head with Pytea (N=34, McNemar exact p=0.00195, paired-bootstrap CI [+14.7, +44.1] pp) is a methodologically sound way of isolating the catalogue-confound from the comparison.
- Pre-registration of the post-freeze GitHub-search query (2026-04-08) and the freeze-hash protocol (`040f6f3`) credibly rules out retro-fitting handlers to the headline corpus.
- A genuine ablation/falsification surface exists: Table 4 shows TG and FakeTensorMode are complementary (20 vs 5) on importable inputs, and Table 5 honestly reports that two of five claimed knobs (CEGAR, phase) are no-ops in the current implementation.

## Weaknesses
- The headline post-freeze unfiltered comparison (Table 3: TG 5/15 vs FakeTensor 2/15 vs Pytea 3/15) is not statistically separable at α=0.05 (Fisher p=0.39 and 0.68; the authors say so), and the Bayes factors (8.1, 3.6) do not clear "strong evidence". Yet the abstract still presents this as "point estimate above both baselines". On N=15 with these p-values, the substantive evidence for TG > baselines on out-of-distribution PRs is weak; a larger pre-registered draw (N≥50, same query) is needed before this can be the third headline number.
- Theorem 2's soundness scope (Lean-audited + pen-and-paper handlers, i.e. 28+3=31/79) covers only 36/185 in-soundness verdicts on the 488-block corpus per the paper itself (line ~498); the remaining 105/185 verdicts touch at least one tested-only handler and are therefore outside the formal soundness theorem. The contribution as advertised in the abstract ("Lean-audited operator-rule table") understates that 48/79 handlers are tested-only and most real-corpus verdicts depend on them.
- Theorem 3 (compositional soundness of the assume/guarantee discipline) is mechanised only on a 3-operator DSL (matmul/view/add), yet C2 is sold as a primary contribution. The gap between "assume/guarantee at the `nn.Module` boundary" as a research artifact and what is actually proven sound is substantial; please either mechanise the composition rule on a non-trivial subset of the 79-handler surface or restate C2 to scope the formal claim to the 3-operator DSL throughout the contribution list, not just in the theorem statement.
- The bug-corpus exclusion pipeline removes ~237/1087 (~22%) of keyword hits under categories (iii) distributed-shape and (iv) config-attribute, where the latter is precisely the silent-miss class flagged in §6 and produces 0/113 RP in the reported audit. The 53/60 (88.3%) headline therefore measures performance conditional on the fragment, not on shape-bug PRs in the wild. A complementary "headline-on-all-1087" or "headline-on-(60+113)" denominator would let readers price in the scope cap.
- The Theorem 5 instantiation uses a "documented forward-signature surrogate" on 4 of the 14 modules (and on 16 of the 17 modules in the prior audit because "full instantiation exceeds end-to-end constraint solving"). The empirical falsifier (zero in-contract recompiles outside the catalogue) is therefore evaluated against a hand-written contract, not against the contract TG would synthesise end-to-end on a transformer. Either run the audit end-to-end on at least one transformer block, or drop "Dynamo-guard correspondence" from the contributions and frame Theorem 5 as a CNN-block result with an open transformer obligation.
- The localisation result (Sec 4.2: tracer ±5 lines on 14/17 caught bugs) has a self-selecting numerator: the 13/30 cases where TG did not refute or returned `location.line=0` are exactly the cases where localisation is needed and missing. The reported 82% should be paired with an end-to-end "fraction of the 30 bugs localised within ±5 lines" (i.e. 14/30 = 47%) so the headline is comparable to a runtime tracer that always fires.
- The L1 (CEGAR) and L3 (phase) features ship in the analyser but are no-ops (Table 5 caption), which is honest but raises a TCB concern: dead-but-present code paths can still affect verdict computation. Please confirm (e.g. by running the test suite with these modules deleted, not just disabled) that the unused CEGAR loop and the always-satisfiable phase encoder cannot influence any RP/CV verdict on the headline corpora.
- The grad-flag silent-misclassification regime (parameter sharing under renamed attributes; ≤12% prevalence) is acknowledged but not reflected in the headline. Since C3 explicitly claims "0/50 false positives" on the backward verifier, please report the false-negative rate of the silent-misclassification regime on a corpus drawn from the ≤12% slice (e.g. 50 training scripts that exhibit tied weights), so C3's calibration matches its denominator.

## Questions
- On the post-freeze N=15 sample, what is the smallest N at which the pre-registered query would give Fisher-exact p<0.05 vs both baselines under the observed 5/15, 2/15, 3/15 rates? Is a pre-registered N=50 draw feasible before camera-ready?
- How many of the 53/60 RPs on the historical bug corpus depend on at least one handler outside the 31 Lean-audited + pen-and-paper soundness scope? A per-bug attribution would let readers compute the in-soundness RP count directly.
- For Theorem 5's necessary-direction claim, does the 8.8% (48/544) in-contract recompile rate include any guards on shape/dtype/rank variables that are in `catalogue(M)` syntactically but whose specialiser bit is read on a different aten op than the one TG's rule fires on? The falsification predicate as stated would not catch this aliasing.
- Can the assume/guarantee composition rule (Theorem 3) be mechanised on at least the matmul/view/add/broadcast/conv2d/reshape subset, or is there a structural obstacle (e.g. handler representation in Python that does not lift to Lean)? If so, please name it.
- The 32/34 vs 22/34 fragment-fair Pytea comparison enforces the modern subset by a static AST predicate over Pytea's commit `cb02a8a` catalogue; what is the verdict triple if this predicate is also enforced on TG's *handler dispatch trace* (not just the AST), to ensure no TG handler outside Pytea's catalogue contributed to a refutation indirectly via Z3?
- For the 2/10 silent verifieds on `rb_001`/`rb_002` (semantically-aliased view bugs), is there an einops-style typed-target extension that the existing rule table can encode, or does this require a new judgement form? A one-paragraph sketch in the limitations would clarify whether this is a future-rule or future-fragment problem.

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 3
Overall: 6

## Borderline reasons
The single change that would push my Overall to 7 is a pre-registered N≥50 post-freeze draw under the same GitHub-search query showing TG > FakeTensor and TG > Pytea at Fisher-exact p<0.05; that would convert the third headline number from a directional point estimate to a real claim, and the rest of the paper (Lean audit, fragment-fair Pytea win, calibrated reporting) is already strong enough to carry an accept at that point.

Round: 1


Changes   +0 -0
Requests  7.5 Premium (2m 5s)
Tokens    ↑ 564.6k • ↓ 6.2k • 515.4k (cached)
