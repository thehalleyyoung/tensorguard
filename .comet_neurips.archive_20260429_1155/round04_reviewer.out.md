● Read round04_reviewer.md
  │ .comet_neurips/round04_reviewer.md
  └ 72 lines read

● List repo and comet_neurips (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ && ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguar
  │ d/.comet_neurips/ 2>/dev/null
  └ 18 lines...

● Count lines and list key dirs (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l neurips.tex neurips.txt README.md && ls src/ benchmarks/
  │ real_benchmarks/ tests/ 2>/dev/null | head -100
  └ 105 lines...

● Read neurips.tex
  └ 93 lines read

● List sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls sections_v5/ && wc -l sections_v5/*.tex
  └ 2 lines...

● find tex files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && find . -path ./.git -prune -o -name "*.tex" -print 2>/dev/null | head -50
  └ 51 lines...

● Read neurips.txt
  └ L1:200 (200 lines read)

● Read neurips.txt
  └ L200:600 (401 lines read)

● Read neurips.txt
  └ L600:1100 (501 lines read)

● Read neurips.txt
  └ L1100:1600 (501 lines read)

● Read neurips.txt
  └ L1600:2100 (501 lines read)

● prior reviews (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/.
  │ comet_neurips && head -50 round03_reviewer.out.md && echo "===" && head -100
  │ round02_reviewer.out.md
  └ 152 lines...

## Summary
TensorGuard is presented as a sound, no-execution refinement-type checker for PyTorch `nn.Module` forward methods, with tensor types `Tensor{s,g | φ}` carrying a symbolic shape, a static gradient-required flag, and a Z3-decidable predicate, composed across class boundaries by an assume/guarantee discipline and supplemented by a backward verifier for three silent-zero-grad bug classes. A one-directional Dynamo-guard correspondence (Theorem 5) is established, and a Lean 4 audit closes 11/11 previously-axiomatic soundness lemmas sorry-free over 28 of 79 shape-transfer handlers, with bit-for-bit agreement against `torch 2.9.1` on 28k in-fragment samples. Empirically the system reports 53/60 RP on a curated historical bug corpus, 32/34 vs Pytea 22/34 on a 2022-catalogue-restricted modern subset (McNemar p=0.00195), 7/10 RP@0.99 on upstream-faithful re-extracts of 10 public bug-PRs, and 5/15 catches on an unfiltered post-freeze sample (vs FakeTensor 2/15, Pytea 3/15; gap not separable at α=0.05). On a 488-block real-source corpus the user-visible verdict is `34 V / 0 RP / 206 LW / 248 A`, openly disclosed as a coverage measurement rather than a bug-finding result.

## Strengths
- The verdict taxonomy (V / RP / CV / LW / Abstain) and the explicit scoping of Theorem 2 to RP+CV only is unusually well-calibrated; the paper does not silently inflate `Verified` with `LW` (Sec. 4.1, Table 1). The user-visible recomputation (`34 V / 0 RP / 206 LW / 248 A` under free-symbolic config) is reported in the body, not buried.
- The Lean 4 audit is genuinely non-trivial: 11/11 previously-axiomatic lemmas closed sorry-free, the `lean/TensorGuard/` tree builds under `lake build`, and the rules are bit-mirrored against `torch 2.9.1` on 28,000/28,000 in-envelope samples plus a ~2,400-sample off-envelope boundary check (Sec. 4.4).
- The Pytea head-to-head includes verification-time fragment-fairness enforcement on both sides (`verify_modern_subset_enforced.py`), silent-skip exclusion of 3 nominal `Verified` rows, and a paired-bootstrap CI [+14.7, +44.1] pp around the +29.4 pp point estimate (Sec. 4.1).
- The hybrid stress table (Table 4, N=25) is a real falsifier of the "TG-only catches everything FT misses" reading: TG-only 20, FT-only 5, surfaces complementary on importable inputs.

## Weaknesses
- **The mechanised soundness perimeter is far smaller than the title advertises.** Table 7 maps the 79 handlers to {28 Lean-audited, 3 pen-and-paper, 48 tested-only}. On the 488-block corpus only 36/185 in-soundness verdicts (11 of 57 V, 25 of 128 CV) touch *only* in-scope handlers, while 105/185 touch at least one tested-only handler (Sec. 4.4, "Calibrated framing"). Theorem 3 (assume/guarantee composition) is mechanised on a 3-operator DSL (matmul/view/add). The title — "Sound Static Verification … with a … Lean-Audited Operator-Rule Table" — is technically defensible but materially over-sells: most verdicts the paper reports rely on handlers outside the Lean perimeter. A title or abstract that names the audited fraction (e.g. "28/79-handler") would be honest.
- **The 60-bug curation excludes ~22% of the funnel (124 distributed + 113 config-attribute hits, Sec. 4.1 "Bug-corpus exclusion denominator") and the 113 config-attribute slice is exactly the regime the paper sells as TG's central advantage** ("symbolic config front-end binds `self.config.<attr>` to fresh SymInts so that HuggingFace-style modules … can be checked once and reused", Sec. 3.1). Dropping the config-attribute bugs from the *bug* denominator while keeping the config-envelope feature in the *contribution* list is double-booking. The headline 53/60 (88.3%) and the Wilson CI [77.8%, 94.2%] are conditional on this exclusion. The paper should report the RP rate on the 113-bug config-attribute slice as an independent number, not just the post-freeze 5/15.
- **The Pytea baseline is pinned to a 2022 catalogue (commit `cb02a8a`, 2022-04-26).** The fair-head-to-head subset is, by construction, "operators in Pytea's 2022 catalogue" (Sec. 4.1). Pytea has no SDPA, RMSNorm, modern attention, or einsum handler, and the paper itself acknowledges that 23/29 Pytea misses on the 60-bug corpus are "pure operator-catalogue gaps". The McNemar p=0.00195 therefore quantifies a four-year catalogue gap as much as a methodological advance. A Pytea fork with handlers ported (or symmetrically: TG restricted to *Pytea's* ops on the modern-attention bugs and *re-run with Pytea's* solver) is the missing comparator. Without it the headline gap is not identified.
- **The strongest external-validity result reverses the headline.** On the unfiltered pre-registered post-freeze sample (Table 3, N=15), TG catches 5/15, FakeTensor 2/15, Pytea 3/15, with two-sided Fisher-exact p=0.39 (vs FT) and p=0.68 (vs Pytea); the abstract's confidence interval [15.2%, 58.3%] crosses both baselines. The abstract sells 88.3% on a curated corpus and 32/34 on a 2022-restricted subset, but the only sample drawn without filtering for fragment fit shows TG statistically indistinguishable from execution-based baselines. The abstract should lead with the unfiltered measurement and treat the curated numbers as supporting, not the reverse.
- **CEGAR contract discovery and the train/eval phase check are admitted no-ops on every reported real corpus.** Table 5 caption: L1 (CEGAR) and L3 (phase) "did not discriminate any case"; Sec. 4.2 "Real-corpus ablation": "none of CEGAR, device-flag, phase, grad-flow, or low-conf gating discriminates on the real bugs". The Implementation section still describes them as live components, and the original contribution list (C5) historically named CEGAR as an architectural pillar. Either remove the no-op components from the architecture description (Sec. 3) and restate the contribution as a 3-feature analyser, or supply a real-bug case on which CEGAR fires.
- **Localisation: 33/33 within ±5 lines is uncalibrated.** Sec. 4.2 itself flags that the AST-walk strategy and the heuristic ground truth "share information for the patterns where the marker was unavailable"; only 3/3 marker-only items are independently scored. A 3-item denominator does not support a localisation claim, and the 33/33 figure should be removed from any summary until a ≥30-item marker-only audit is run.
- **Theorem 5 (Dynamo-guard correspondence) is essentially never instantiated end-to-end.** Sec. 4.3 admits that on 16 of the 17 audited modules "the audit takes the documented `forward` signature as the contract because the full instantiated module exceeds end-to-end constraint solving"; the actual end-to-end TG-verified instances are 5 (`BasicBlock`, `Bottleneck`, `InvertedResidual`, `Fire`, ViT `Block`). The "calibration … rather than … in-theorem instantiation" disclaimer is appropriate, but a theorem with five end-to-end witnesses is a preliminary result, not a labelled contribution (C4). The 48/544 in-contract recompiles are also all bucket `INT`, which the theorem already excludes by construction; this is consistency, not evidence.
- **The paper as written reads in many places as a revision diary** — phrases like "(round-7 W5)", "(Round-2 reviewer Q1)", "(round-3 borderline lift)", "(round-5 Q4)", "round-3 Q6", "the rb_pf_001 flip from silent to RP@0.99 this round", "the round-7 int(...)/float(...)-cast fold introduced this round" appear throughout Sec. 4.1, Sec. 4.4, and Sec. 6. A final NeurIPS submission should not narrate its own revision history; please either remove or rewrite these passages so the empirical claims read as claims rather than as responses to prior reviews.

## Questions
- What is TG's RP rate on the 113-bug "config-attribute" slice excluded under exclusion rule (iv) of the bug-corpus funnel (Sec. 4.1, "Bug-corpus exclusion denominator")? This is the regime the symbolic-config front-end is sold against.
- Can you provide TG vs Pytea on the modern-attention bugs after either (a) porting SDPA/RMSNorm/einsum handlers into Pytea's Z3 frontend, or (b) restricting *both* tools to Pytea's 2022 catalogue and running the *same* set of bugs on each? The current asymmetry conflates four years of operator coverage with a methodological contribution.
- Of the 488 real-source blocks, can you exhibit a single block (not a hand-distilled or upstream-faithful re-extract) on which TG returns `Refuted-Proof` *under the user-visible free-symbolic-config regime*? If not, the abstract should be reworded to scope the unconditional-RP claim to the bug corpora.
- What is the marker-only localisation rate on a fresh ≥30-item audit where the AST-walk and the ground-truth marker do not share information? The current 33/33 number is dependent on shared information by the paper's own admission.
- For Theorem 5, can you re-run the end-to-end (no `forward`-signature surrogate) audit on at least 15 modules — including at least three transformer blocks where the in-contract recompile rate currently relies on the surrogate — and report `{SHAPE, DTYPE, RANK, INT}` recompile counts per module?
- The first-order grad-flag lattice silently misclassifies parameter-sharing-under-renamed-attribute, with prevalence ≤12% of training scripts (Sec. 6). What is TG's silent-error rate on, e.g., the 17 real torchvision/HF/timm modules audited for Theorem 5, and on a stratified sample restricted to scripts that pass the renamed-attribute filter (i.e. the population where the bug actually fires)?

## Scores
Soundness: 3
Presentation: 2
Contribution: 3
Confidence: 4
Overall: 4

## Borderline reasons
The single change that would push the overall score up by one point is to replace the curated 60-bug headline (88.3%, and the Pytea-2022 32/34) with the unfiltered post-freeze N=15 result and an RP rate on the 113-bug config-attribute slice as the lead numbers, *and* to either drop CEGAR/phase from the Implementation section or report a real-bug case on which they fire. Without these, the abstract continues to advertise figures that the paper's own most-external measurements undercut. Removing the round-by-round revision narration in §4.1/§4.4/§6 and re-running the Pytea comparison on a non-frozen catalogue would together lift my score further.


Changes   +0 -0
Requests  7.5 Premium (2m 18s)
Tokens    ↑ 596.7k • ↓ 6.4k • 534.3k (cached)
