● Read round19_reviewer.md
  │ .comet_neurips/round19_reviewer.md
  └ 118 lines read

● list paper (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/ 2>/dev/null | head -50; echo ---; ls /Users/halleyyoung/Documents
  │ /div/mathdivergence/halley-labs/tensorguard/docs/paper/sections_v5/
  │ 2>/dev/null
  └ 63 lines...

● view paper text (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l neurips.txt 2>/dev/null; head -200 neurips.txt 2>/dev/null
  └ 202 lines...

✗ search prior weakness areas (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "post-freeze\|N=15\|N
  │ =15\|Benjamini\|Fisher\|p=0.39\|62/185\|66/185\|128\|128≠\|stale\|56
  │ refutations\|all 56\|R=53\|7/7\|HuggingFace\|natural\|kill
  │ rate\|7/50\|14%\|targeted\|conv2d 53\|einsum 100" neurips.txt | head -80
  └ Command not executed. The 'kill' command must specify at least one numeric
    PID. Usage: kill <PID> or kill -9 <PID>

● grep prior weakness areas (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -nE "post-freeze|N=15|N
  │ =15|Benjamini|Fisher|p=0\.39|62/185|66/185|128|stale|56 refutations|all
  │ 56|R=53|7/7|HuggingFace|natural|7/50|conv2d 53|einsum 100|mutation"
  │ neurips.txt | head -80
  └ 81 lines...

● Read neurips.txt
  └ L1280:1430 (151 lines read)

● find handler scope numbers (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -nE "62/185|66/185|/185|185|in-soundness|tested-only|footprint"
  │ neurips.txt | head -40
  └ 23 lines...

● Read neurips.txt
  └ L550:720 (171 lines read)

● find caption text (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper && grep -n "all 5[0-9] refutations\|56 refutations\|53
  │ refutations\|R = 53\|Refuted-Proof" sections_v5/*.tex 2>/dev/null | head -30
  └ 31 lines...

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L400:470 (71 lines read)

● Read neurips.txt
  └ L1900:2000 (101 lines read)

Now I have enough context to write the review.

## Summary
TensorGuard is a no-execution, refinement-type checker for PyTorch nn.Module forward methods that statically verifies tensor shapes and gradient flow without instantiating or tracing the module. The paper contributes (i) a refinement-type calculus Tensor{s,g|φ} discharged via Z3, (ii) an assume/guarantee composition rule for the class boundary, mechanised in Lean 4 over a 17-operator DSL (28 of 79 handlers Lean-audited, 7 pen-and-paper, 44 tested-only), (iii) an autograd-aware backward verifier targeting silent-zero-grad bugs, and (iv) an exploratory necessary-direction Dynamo-guard inclusion lemma. Empirically the paper reports 53/60 RP on a curated bug corpus, 32/34 vs Pytea 25/34 on the fragment-fair head-to-head (McNemar p=0.0156), 0 unconditional RP on the 488-block real-source corpus (calibrated as a coverage measurement), 7/7 on naturally-extracted HuggingFace decoder bugs, and a pre-registered post-freeze N=15 sample where TG catches 5/15 (vs FakeTensorMode 2/15, Pytea 3/15) but pairwise Fisher-exact and BH-corrected p-values are non-significant.

## Prior weakness disposition
- [PARTIAL] The N=15 post-freeze test is the only pre-registered unbiased generalisation test, and after Benjamini–Hochberg correction all three pairwise Fisher-exact p-values adjust to 1.00 -- The paper now explicitly reports the BH-adjusted 1.00 values, Wilson CIs, and a power calculation locating N_new=26–77 for significance, but the only pre-registered unbiased generalisation test is still statistically null and is now framed as "calibrated confidence, not significance" rather than separated.
- [PARTIAL] The handler-scope arithmetic is inconsistent: 62+66=128≠185 -- The abstract now says "11/57 Verified and 25/128 CV touch only the audited footprint, remaining 103/185 in-soundness verdicts touching at least one tested-only handler," but Section 4.4 reports the same decomposition as 36/185 audited-only and 33+72=105/185 tested-only; abstract (103) and body (105) still disagree, and 36+103=139≠185 in the abstract.
- [RESOLVED] Table 1's caption states "all 56 refutations are Refuted-Proof" while the table body shows R=53 -- The eval_v6.tex caption now reads "all 53 refutations are Refuted-Proof," matching the body and abstract.
- [RESOLVED] The 7/7 natural HuggingFace bugs are presented without a torch.compile baseline -- The paper now reports torch.compile (fullgraph=True, dynamic=False) and FakeTensorMode also raise on 7/7 of the reduced repros, and explicitly recasts the TG advantage as firing on the un-instantiable original class source rather than a head-to-head margin on the reduced modules.
- [PARTIAL] The global mutation kill rate (7/50, 14%) remains the headline; the targeted per-handler improvement (conv2d 53%, einsum 100%) is on a corpus constructed for that purpose -- The paper now keeps the 14% multi-corpus rate as the analyser-wide headline and presents the 53%/100% targeted figures only as per-handler load-bearing measurements on an admittedly purpose-built 18-case extension; the criticism that the headline analyser-wide robustness number is still 14% has not been moved.

## Strengths
- Genuine and unusual artefact: a refinement-type checker that operates on raw class source without constructor instantiation, plus a Lean 4 mechanisation of the composition rule and 11/11 previously-axiomatic shape-soundness lemmas closed sorry-free; 28,000/28,000 random-agreement samples against torch 2.9.1.
- Calibrated reporting: every (tool, input) cell carries a verdict from a fixed taxonomy; the analyser concedes "0 unconditional RP" on the 488-block real-source corpus and frames it as a fragment-coverage measurement, which is rare and admirable in this literature.
- The fragment-fair N=34 McNemar comparison against Pytea (b=7, c=0, exact two-sided p=0.0156) is a properly paired statistical test on the matched modern subset.
- Engineering breadth: backward verifier with 500/500 static↔runtime agreement and 0/50 false positives, 488-block content-addressed corpus, AST diversity K_ast=406, hybrid mode, mutation testing with per-mutant code-site classification.

## Weaknesses
- The handler-scope arithmetic in §4.4 and the abstract still does not reconcile. Abstract says "11/57 Verified and 25/128 CV touch only the audited footprint, with the remaining 103/185 in-soundness verdicts touching at least one of the 44 tested-only handlers." 11+25=36 audited-only, so the remainder over 185 is 149, not 103. Body §4.4 partitions the same 185 as 36 audited-only + 33+72=105 tested-only, leaving 44 unaccounted-for. Abstract (103) and body (105) disagree, and neither partition sums to 185. This is the headline soundness-footprint number and must be a clean partition.
- The only pre-registered unbiased generalisation test (N=15 unfiltered post-freeze) remains statistically null after BH correction (all adjusted p=1.00), and the paper's own power calculation says ~26–77 additional samples are needed to reach α=0.05 on either pair. The paper currently uses language like "TG strictly above the two execution-based baselines" on the basis of point estimates 5/15 vs 2/15 and 3/15. A 5/15 vs 2/15 contrast on N=15 with overlapping Wilson CIs ([15.2%,58.3%] vs [3.7%,37.9%]) does not support "strictly above." The abstract should either drop the comparative framing on this corpus or the corpus should be extended to the pre-registered second-wave size.
- The headline analyser-wide mutation-kill rate is 7/50 (14%) on the union of three corpora. The two zero-kill load-bearing handlers (conv2d 0/10, einsum 0/10) on the 60-bug regression corpus are then resurrected by an 18-case targeted extension corpus constructed specifically to exercise their arithmetic, where they reach 53% and 100%. This is a circular measurement; the per-handler numbers cannot be quoted as evidence of robustness at the same level as the 14% multi-corpus number, and the paper's contribution claim (C5) does not currently disclose this construction-of-evaluation issue at the point where the targeted numbers are introduced.
- C4 (Dynamo-guard inclusion, Theorem 5) is empirically instantiated end-to-end without surrogate on 9 CNN blocks but only 1/4 transformer blocks; the paper itself flags this. Given that the entire motivating story in §1 is about HuggingFace transformers that cannot be instantiated, a result that holds end-to-end for 1/4 transformer blocks is too thin to support C4 as a numbered contribution alongside the calculus and the Lean audit. C4 should either be demoted to an appendix discussion or extended to ≥ a half-dozen non-surrogate transformer blocks.
- The 488-block "headline" is "0 unconditional RP," with all 206 refutations being CV (synthesised caller-rely) or LW. The 92.2% (118/128) joint-realisability of the synthesised assume_M is reported, but the headline therefore says nothing about whether TG would catch real bugs in real callers on this corpus, only that its assumes are satisfiable by a default *Config(). There is no reported study that pairs the 128 CV verdicts with the natural caller in the same library and shows that the caller's actual call-site shape would in fact violate the assume; without that, "188/206 = 91.3% caller-realisable" is consistent with TG flagging code that does not bug in practice.
- C6's "28,000/28,000 agree with torch 2.9.1" is sampled "uniformly within the in-fragment envelope" of each rule. The complementary boundary check is reported on only 10 of 28 rules with ~2,400 off-envelope samples. Without boundary coverage on all 28 rules, the agreement headline overstates the audit's discriminative power: it confirms agreement on inputs the rule already says it covers, which is necessary but not sufficient for "the rule table is correct."
- Several of the most consequential numbers in the abstract — 11/57, 25/128, 103/185, 44 tested-only, 28 of 79 handlers — are scattered across the abstract, contributions list, §4.4, and Table 7 in inconsistent partitionings (e.g., contributions reference 28+7=35 handlers in §4.1's footprint discussion of the 6 RP fires, but Table 7's summary line is "28 + 7 + 44 = 79"). The reader cannot reconstruct a single handler-soundness ledger from the paper.

## Questions
- Please reconcile the 488-block soundness-footprint partition into a single explicit table: of the 57 Verified verdicts, how many touch (a) only Lean-audited handlers, (b) only Lean-or-pen-paper-audited handlers, (c) at least one tested-only handler, (d) no handler at all? Same four-way split for the 128 CV verdicts. The four cells should sum to 57 and 128 respectively, and the abstract should quote whichever totals make 11+25=36 plus the remainder add to 185.
- For the post-freeze N=15 corpus: do you intend to run the pre-registered second wave (N_new ≥ 26 by your own power calculation)? If not, please either remove the comparative claim ("strictly above the two execution-based baselines") from the abstract/§4 headline or restrict it to the descriptive statement "5/15 catches vs 2/15 and 3/15, no significant pairwise contrast at α=0.05."
- For C4: can you report end-to-end (non-surrogate) Theorem-5 instantiation on at least four additional transformer blocks (e.g., a Llama, Mistral, Qwen2 and Phi-3 attention/MLP block) so the transformer base is not 1/4? If the constraint solver still does not close, please report that explicitly as a negative result with the offending constraint families.
- For the 128 CV verdicts on the 488-block corpus: of the 118 "joint-realisable under default *Config()" rows, on how many does the natural in-library caller actually violate the assume_M at its real call site? A number here would convert the 92.2% realisability statistic into a witness rate of "TG would catch a real bug in a real caller."
- The mutation-kill targeted corpus (12 buggy + 6 clean modules constructed to cover conv2d/einsum arithmetic paths) is introduced in §4.3 to lift the per-handler numbers off zero. Could you report the per-handler kill rate on the 488-block corpus restricted to forward bodies that exercise conv2d / einsum arithmetic — i.e., a non-purpose-built corpus — and quote that as the load-bearing per-handler robustness?
- Boundary-check coverage of the rule audit: please report the off-envelope agreement test on all 28 Lean-audited rules rather than 10/28, with the same ~2,400 sample budget per rule, so the audit is symmetric.

## Scores
Soundness: 3
Presentation: 2
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would push my score from 5 to 6 is reconciling the handler-scope partition into one explicit, summing-to-185 table that the abstract and §4.4 both quote, combined with either running the pre-registered N≥26 second wave on the post-freeze corpus or removing the "strictly above baselines" comparative framing from the abstract. Either fix individually would also help; both together would move it to a clear borderline accept.


Changes   +0 -0
Requests  7.5 Premium (2m 40s)
Tokens    ↑ 515.4k • ↓ 8.5k • 476.8k (cached)
