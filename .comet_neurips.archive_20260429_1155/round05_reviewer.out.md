● Read round05_reviewer.md
  │ .comet_neurips/round05_reviewer.md
  └ 72 lines read

● List comet_neurips dir (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/.
  │ comet_neurips/ 2>/dev/null | head -50
  └ 25 lines...

● Check paper sizes (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l neurips.tex neurips.txt && ls -la neurips.pdf
  └ 5 lines...

● Read neurips.tex
  └ 93 lines read

● Read round04_reviewer.out.md
  │ .comet_neurips/round04_reviewer.out.md
  └ 91 lines read

● List section files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls sections_v5/ && wc -l sections_v5/*.tex
  └ 2 lines...

● Find sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && find . -name "sections_v5" -type d 2>/dev/null && find . -name "*v6*.tex"
  │ 2>/dev/null | head -20
  └ 18 lines...

● Check paper sizes and prior scores (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l docs/paper/sections_v5/*.tex && cat .comet_neurips/scores.jsonl
  └ 38 lines...

● Read neurips.txt
  └ L1:200 (200 lines read)

● Read neurips.txt
  └ L200:700 (501 lines read)

● Read neurips.txt
  └ L700:1300 (550 lines read)

● Read neurips.txt
  └ L1250:1900 (519 lines read)

● Read neurips.txt
  └ L1769:2100 (332 lines read)

## Summary
TensorGuard is a no-execution refinement-type checker for `nn.Module` forward methods, with tensor types `Tensor{s, g | φ}` carrying symbolic shape, a static gradient-required flag, and Z3-decidable side conditions, composed at the `nn.Module` boundary by an assume/guarantee discipline and supplemented by a backward verifier for three silent-zero-grad bug classes. A Dynamo-guard correspondence (Theorem 5) is asserted in the necessary direction only, and a Lean 4 audit closes 11/11 previously-axiomatic soundness lemmas sorry-free over 28 of 79 handlers, with bit-for-bit agreement against torch 2.9.1 on 28k in-fragment samples. Empirically TG reports 53/60 RP on a curated historical bug corpus and 32/34 vs Pytea 22/34 on a 2022-catalogue-restricted modern subset (McNemar p=0.00195); on the upstream-faithful re-extracts of 10 public bug-PRs, 7/10 RP@0.99 plus 1/10 at 0.80; and on an unfiltered pre-registered N=15 post-freeze sample, 5/15 catches (Wilson [15.2%, 58.3%]) versus FakeTensor 2/15 and Pytea 3/15. On the 488-block real-source corpus the user-visible verdict is `34 V / 0 RP / 206 LW / 248 A`, openly disclosed as a coverage measurement rather than a bug-finding result.

## Strengths
- The verdict taxonomy (V / RP / CV / LW / Abstain) is rigorously scoped: Theorem 2 covers RP+CV only, LW is explicitly outside the soundness perimeter, and the user-visible free-symbolic-config recomputation (`34 V / 0 RP / 206 LW / 248 A`) is in the body rather than buried (Sec. 4.1, Table 1, "Headline. Calibration first.").
- The Lean 4 audit is genuinely non-trivial: 11/11 previously-axiomatic lemmas closed sorry-free, the `lean/TensorGuard/` tree builds under `lake build`, and the rule table is bit-mirrored against `torch 2.9.1` on 28,000/28,000 in-envelope samples plus a ~2,400-sample off-envelope boundary check (Sec. 4.4).
- The Pytea head-to-head includes verification-time fragment-fairness enforcement on both sides (`verify_modern_subset_enforced.py`), exclusion of three nominal-Verified silent skips, and a paired-bootstrap CI [+14.7, +44.1] pp around +29.4 pp (Sec. 4.1).
- The hybrid stress contingency table (Table 4, N=25) is a real falsifier: TG-only 20, FT-only 5 — the surfaces are complementary rather than TG-strictly-dominant on importable inputs.
- The 488-block corpus diversity result (`Kast=406`, `Khandler=345`, largest cluster 20 members, 369/406 singletons; Sec. 4.1 "Block-corpus diversity") rules out the "few dressed-up duplicates" reading.

## Weaknesses
- **The title still oversells the mechanised perimeter.** "Sound Static Verification … with a … Lean-Audited Operator-Rule Table" is paired with Table 7's `28 Lean-audited / 3 pen-and-paper / 48 tested-only` split and with Sec. 4.4's per-block scope table showing 36/185 in-soundness verdicts touch only in-scope handlers, while 105/185 touch at least one tested-only handler. The 48 untested-by-Lean handlers cover the majority of verdicts the paper actually reports. The honest title would name the audited fraction (e.g. "28/79-handler Lean-audited rule table") rather than implying the table as a whole is Lean-audited.
- **The 60-bug curation excludes ~22% of the funnel and the 113 config-attribute slice is exactly the regime the paper sells the symbolic-config front-end against.** Sec. 4.1 "Bug-corpus exclusion denominator" confirms 113/1087 hits dropped under exclusion (iv), and Sec. 3.1 markets the symbolic config environment as "binds `self.config.<attr>` to fresh SymInts so HuggingFace-style modules ... can be checked once and reused". The headline 53/60 (88.3%) is therefore conditional on dropping the very bugs that exercise the headline feature. Round 4 already asked for the RP rate on the 113-bug config-attribute slice as an independent number; the current revision still does not report it.
- **The Pytea baseline is pinned to a 2022 catalogue (commit `cb02a8a`, 2022-04-26).** The fair-head-to-head is by construction "operators in Pytea's 2022 catalogue" (Sec. 4.1). The paper itself notes 23/29 Pytea misses on the 60-bug corpus are "pure operator-catalogue gaps" (no SDPA, RMSNorm, einsum, Conv1d/3d, GroupNorm, BatchNorm1d). McNemar p=0.00195 thus quantifies four years of operator coverage as much as a methodological advance. The principled comparator — Pytea forked with SDPA/RMSNorm/einsum handlers ported, or symmetrically TG and Pytea both restricted to Pytea's 2022 catalogue on the modern-attention bugs — is not run.
- **The unfiltered post-freeze sample (Table 3, N=15) reverses the headline.** TG 5/15, FakeTensor 2/15, Pytea 3/15, with two-sided Fisher-exact p=0.39 vs FT and p=0.68 vs Pytea, and the abstract's interval [15.2%, 58.3%] crosses both baselines. The abstract leads with 88.3% on the curated corpus and 32/34 on the 2022-restricted subset; the only sample drawn without filtering for fragment fit shows TG statistically indistinguishable from execution-based baselines. The lead number should be the unfiltered measurement, with curated numbers as supporting.
- **CEGAR contract discovery and the train/eval phase check are admitted no-ops on every reported real corpus.** Table 5 caption: L1 (CEGAR) and L3 (phase) "did not discriminate any case in the current implementation." Sec. 4.2 "Real-corpus ablation": "none of CEGAR, device-flag, phase, grad-flow, or low-conf gating discriminates on the real bugs." The Implementation section still describes them as live components and the contribution list historically named CEGAR as an architectural pillar. Either remove the no-op components from the architecture description or supply a real-bug case on which CEGAR fires.
- **Localisation: the 33/33 within ±5 lines is uncalibrated.** Sec. 4.2 itself flags that "the heuristic ground truth and the AST-walk strategy share information for the patterns where the marker was unavailable"; only 3/3 marker-only items are independently scored. A 3-item denominator does not support a localisation claim, and the 33/33 figure should be removed from any summary text until a ≥30-item marker-only audit is run.
- **Theorem 5 is essentially never instantiated end-to-end.** Sec. 4.3 admits that on 16 of the 17 audited modules "the audit takes the documented forward signature as the contract because the full instantiated module exceeds end-to-end constraint solving"; the actual end-to-end TG-verified instances are 5 (`BasicBlock`, `Bottleneck`, `InvertedResidual`, `Fire`, ViT `Block`). The 48/544 in-contract recompiles are *all* bucket `INT`, which the theorem already excludes by construction; this is internal consistency, not evidence. A theorem with five end-to-end witnesses, none of which are large transformer blocks, is a preliminary result, not a labelled contribution (C4).
- **The paper continues to read in many places as a revision diary.** Tags such as "(round-7 W5)" (Sec. 4.3), "(round-2 Q4)" and "(round-7 Q6)" and "(round-5 Q3)" and "(round-3 Q4)" (Sec. 4.1), "(Round-2 reviewer Q1)" and "(Round-2 reviewer Q2)" and "(Round-3 borderline lift)" and "(round-5 Q2; revised round 1)" and "(round-3 Q6)" (Sec. 4.1), and narrative phrases such as "the rb_pf_001 flip from silent to RP@0.99 this round closes the intermediate ... bottleneck reported in round 7" (Sec. 4.1 post-freeze paragraph) and "the round-7 int(...)/float(...)-cast fold introduced this round" (Sec. 6) are present throughout. A NeurIPS submission should read as a self-contained artefact; please rewrite these passages so the empirical claims read as claims rather than as responses to prior reviewers.

## Questions
- What is TG's RP rate on the 113-bug "config-attribute" slice excluded under exclusion rule (iv) (Sec. 4.1, "Bug-corpus exclusion denominator")? Until this is reported, the 53/60 headline cannot be read as evidence about the regime the symbolic-config front-end is sold for.
- Can you provide TG vs Pytea on the modern-attention bugs after either (a) porting SDPA/RMSNorm/einsum handlers into Pytea's Z3 frontend, or (b) restricting *both* tools to Pytea's 2022 catalogue on the same set of bugs? The current asymmetry conflates four years of operator coverage with the methodological contribution.
- Of the 488 real-source blocks, can you exhibit a single block (not a hand-distilled or upstream-faithful re-extract) on which TG returns Refuted-Proof under the user-visible free-symbolic-config regime? If not, the abstract's "0 unconditional Refuted-Proof verdicts" should be the lead, not the 53/60 carry-over.
- What is the marker-only localisation rate on a fresh ≥30-item audit where the AST-walk and the ground-truth marker do not share information? The current 33/33 number is dependent on shared information by the paper's own admission.
- For Theorem 5, can you re-run the end-to-end (no `forward`-signature surrogate) audit on at least 15 modules — including at least three transformer blocks where the in-contract recompile rate currently relies on the surrogate — and report `{SHAPE, DTYPE, RANK, INT}` recompile counts per module?
- The first-order grad-flag lattice silently misclassifies parameter-sharing-under-renamed-attribute, prevalence ≤12% of training scripts (Sec. 6). What is TG's silent-error rate on the 17 real torchvision/HF/timm modules used for Theorem 5, and on a stratified sample restricted to scripts that pass the renamed-attribute filter (i.e. the population where the bug actually fires)?

## Scores
Soundness: 3
Presentation: 2
Contribution: 3
Confidence: 4
Overall: 4

## Borderline reasons
The single change that would push the overall score up by one point is to lead the abstract with the unfiltered post-freeze N=15 result and an RP rate on the 113-bug config-attribute slice, demoting the curated 53/60 and Pytea-2022 32/34 to supporting evidence; combined with either dropping CEGAR/phase from the Implementation section or supplying a real-bug case on which they fire. Removing the round-by-round revision narration in §4.1/§4.3/§6 and re-running the Pytea comparison on a non-frozen catalogue would together lift my score further. Without these, the abstract advertises numbers that the paper's own most-external measurements undercut.


Changes   +0 -0
Requests  7.5 Premium (2m 38s)
Tokens    ↑ 598.4k • ↓ 6.7k • 536.5k (cached)
