● Read round01_reviewer.md
  │ .comet_neurips/round01_reviewer.md
  └ 113 lines read

● List files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls && wc -l neurips.txt neurips.tex 2>/dev/null
  └ 16 lines...

● Read neurips.tex
  └ 87 lines read

● Read neurips.txt
  └ L1:200 (200 lines read)

● Read neurips.txt
  └ L200:600 (401 lines read)

● Read neurips.txt
  └ L600:1100 (501 lines read)

● Read neurips.txt
  └ L1100:1700 (542 lines read)

● Read neurips.txt
  └ L1700:2300 (601 lines read)

● Read neurips.txt
  └ L2300:2900 (601 lines read)

● Inspect repo (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/TensorGuard/ 2>/dev/null | head -30 && echo "---" && ls
  │ reproducibility/ 2>/dev/null | head -40 && echo "---" && ls experiments_v5/
  │ 2>/dev/null | head -30
  └ 78 lines...

## Summary
TensorGuard is a static, no-execution refinement-type checker for PyTorch `nn.Module` forward methods that reasons about symbolic shapes and a flat first-order grad-flag lattice (`{has_grad, no_grad, ⊤}`), discharging side conditions to Z3. The system reports a five-way verdict taxonomy (`Verified / Refuted-Proof / Contract-Violation / Library-Warn / Abstain`); only RP and CV are covered by the soundness theorem. Empirical claims include 53/60 RP on a curated historical bug corpus, 32/34 vs. Pytea 22/34 on a fragment-fair modern subset (McNemar p=0.00195), 5/15 catches on an unfiltered pre-registered post-freeze N=15 PR sample (vs. FakeTensorMode 2/15 and Pytea 3/15, not statistically separable), 0 unconditional RP on the 488 real-source blocks under a free-symbolic regime, and a Lean 4 audit covering 28 of 79 shape-transfer handlers with 11/11 previously-axiomatic soundness lemmas closed sorry-free. A necessary-direction Dynamo-guard correspondence (Thm. 5) is stated and empirically audited on 17 modules.

## Prior weakness disposition
(none — first round)

## Strengths
- Calibration discipline is unusually high for an applied-PL paper: verdicts are partitioned into `RP / CV / LW / Abstain`, the soundness theorem (Thm. 2) is explicitly scoped to RP+CV, the 488-block corpus is honestly reported as 0 unconditional RP under the free-symbolic regime, and the Pytea modern-subset comparison enforces fragment fairness at verification time (`experiments_v5/v8/verify_modern_subset_enforced.py`).
- The Lean 4 audit (`lean/TensorGuard/`) is concrete and inspectable: 28 operator rules with 11 soundness lemmas closed sorry-free, plus a 28,000/28,000 byte-mirror cross-check against `torch 2.9.1`. The honest restatement of `permList_compose` to its in-range form (Sec. I) is the right move, not papered over.
- The pre-registered post-freeze N=15 PR sample, with the freeze hash recorded and a query frozen one day after the catalogue freeze, is a genuine generalisation test rather than a retro-fit, and the off-axis fire (`rb_uf_010`) is correctly counted as a false positive.
- The semantic-aliased view-bug residual (`rb_001`, `rb_002`) is diagnosed precisely (buggy/correct view targets agree on element count for the supplied shape) rather than waved away, and the constructor-bound integer-attribute envelope is reported as a residual, not closed by hand-waving.

## Weaknesses
- The unconditional-RP headline rests almost entirely on a curated 60-bug historical corpus (53/60). On the 488-block real-source corpus the user-visible regime returns **zero** unconditional RP verdicts; on the unfiltered post-freeze N=15 sample the catch rate is 5/15 with Wilson CI [15.2%, 58.3%] and Fisher p=0.39 vs. FakeTensorMode. The headline claim "catches real PyTorch shape bugs that execution-based tools cannot" is therefore not statistically separable from the baselines on the only unfiltered, pre-registered evaluation. Provide either a substantially larger pre-registered post-freeze sample (e.g. N≥60) so the head-to-head Fisher comparison clears α=0.05, or restate the contribution as a calibrated-coverage result rather than an empirical superiority claim.
- The title advertises "Sound Static Verification … with a 28/79-Handler Lean-Audited … Calculus", but only 28/79 ≈ 35% of handlers are Lean-audited, the analyser implementation, AST extractor, backward verifier, Z3 dispatch, and assume/guarantee composition (Thm. 3 is mechanised on a 3-operator DSL only) are *not* mechanised. The phrase "sound static verification" in §1 and the abstract is therefore overloaded relative to what the artefact actually certifies. Either reduce the soundness claim in the title/abstract to the rule-table layer, or extend the Lean audit to the remaining 51 handlers and the assume/guarantee rule on the full operator surface.
- The 488-block "0 unconditional RP / 57 Verified" headline depends critically on the synthesised caller-rely `assume_M`. The CV-witness audit cites 26/128 empty assumes, 90/128 reducing to documented config attributes, and 12/128 PreTrainedModel stubs, but does not show that any of the 90 documented-config assumes is *jointly* satisfied by a real published-checkpoint config (only that the symbols exist in some config). Show, for the full 128-CV set or a uniformly random subsample with a stated CI, that each `assume_M` is satisfied by at least one concrete published-checkpoint instantiation, with the resulting ratio reported.
- Theorem 5 is necessary-direction-only and the empirical audit reports an 8.8% in-contract recompile rate; the falsification predicate (`{SHAPE,DTYPE,RANK} guards outside catalogue`) is exercised on only 17 modules and 48 in-contract recompiles, all of which fall in the `INT` bucket the theorem already excludes. This is not strong evidence for the theorem — it is a "no falsifier observed" result on a small sample. Run the falsifier on a substantially larger module set (e.g. ≥100 timm/HF blocks) and report the resulting fraction; if zero out-of-catalogue SHAPE/DTYPE/RANK guards are seen, that bound is meaningful, otherwise the theorem statement needs to be tightened.
- The per-feature ablation on the real corpus is "a flat line": CEGAR contract discovery and the train/eval phase check are honestly reported as no-ops (`ShapeCEGARLoop` predicates never reach the verdict pipeline; phase encodes `Or(TRAIN, EVAL)` which is always satisfiable). The paper should either remove these from the contribution list (currently still implicitly part of C5) or wire them through to actually fire on at least one real bug.
- The first-order grad-flag lattice `{has_grad, no_grad, ⊤}` is silently incorrect on parameter-sharing-under-renamed-attribute; the prevalence is bounded at ≤12% of training scripts by a self-conducted GitHub sweep but no independent corroboration is given. Either run the backward verifier on a held-out set of HF training scripts containing this construct and report the false-verified rate, or restrict the grad-flow contribution (C3) to non-shared-parameter modules.
- The 33/33 within-±5-line localisation result is, by the paper's own admission, a "consistency check, not a precision claim" because the AST-walk strategy and the heuristic ground truth share information; only 3/3 within-1 on the marker-only items is independent. The marker-only audit needs to be at least the ≥30-item size the paper itself names before any localisation result enters the contributions.
- Presentation: the paper is *exceptionally* dense with rebuttal-style apparatus (round-2 Q4, round-3 Q6, round-5 Q3, round-7 W5, etc.) embedded in the body. The abstract alone is ~22 sentences and mixes the headline with five caveats; Section 4.1 reads as a stack of patches rather than a result section. Restructure §4 so the five "round-N" responses are condensed into a single calibration paragraph; move the per-round disposition to an appendix.

## Questions
- For the N=15 unfiltered post-freeze sample, what is the planned next-round denominator, and will the pre-registration query be the same? On a doubled sample, does the TG-vs-FakeTensorMode Fisher gap clear α=0.05?
- Of the 90/128 "documented-config" CV verdicts, how many are simultaneously satisfied by *the same* published checkpoint config (i.e. the conjunction is realisable in one instantiation), as opposed to each clause being satisfied by some config?
- On the N=10 upstream-faithful re-extracts, what fraction of the verdict path on `rb_003`, `rb_004`, `rb_006`, `rb_010` flows through the round-6 envelope synthesiser additions vs. the legacy v4 path? An ablation that disables only the new constructor-bound integer-attribute envelope on these 4 cases would isolate the round-6 contribution.
- For Theorem 5: in the 17-module audit, were any modules excluded for catalogue mismatch before the recompile counts were taken? If so, what is the recompile rate on the un-pruned set?
- For the 25-block hybrid stress set (Table 4), what is the construction protocol — were the 20 TG-only catches and 5 FT-only catches selected to maximise complementarity? A pre-registration or random-sample protocol would strengthen the "complementary, not coincident" claim.
- The Lean operator-rule table covers 28 of 79 handlers. Which 51 are tested-only, and what is the soundness risk profile (e.g., are the un-audited handlers concentrated in low-frequency operators, or do they include load-bearing rules like SDPA/LayerNorm)?

## Scores
Soundness: 3
Presentation: 2
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would move my overall from 5 to 6 is a larger pre-registered post-freeze sample (N≥60) on which the TG-vs-FakeTensorMode Fisher comparison clears α=0.05 — i.e., turning the calibrated-confidence result of Table 3 into a statistically separated head-to-head on an unfiltered surface. Alternatively, extending the Lean audit to the full 79-handler surface (or to the assume/guarantee rule on more than the 3-operator DSL) would justify the "Sound Static Verification" framing of the title and similarly raise the score.


Changes   +0 -0
Requests  7.5 Premium (1m 53s)
Tokens    ↑ 542.0k • ↓ 5.3k • 474.1k (cached)
