● Read round18_reviewer.md
  │ .comet_neurips/round18_reviewer.md
  └ 119 lines read

● List paper sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls -la docs/paper/sections_v5/ 2>/dev/null | head -40
  └ 35 lines...

● Read paper tex source (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat neurips.tex 2>/dev/null | head -200 || echo "neurips.tex not found"
  └ 95 lines...

● Read evaluation section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat docs/paper/sections_v5/eval_v6.tex
  └ 18 lines...

● Read intro and calculus sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat docs/paper/sections_v5/intro_v6.tex && cat
  │ docs/paper/sections_v5/calculus_v6.tex
  └ 307 lines...

● Read eval section beginning (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -300 /var/folders/2k/n00k2r_d7sb3g3g0jkf059mh0000gn/T/copilot-tool-o
  │ utput-1777530018587-b0vsql.txt
  └ 301 lines...

● Read limitations/conclusion section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat docs/paper/sections_v5/limconc_v6.tex
  └ 193 lines...

● Read appendix (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat docs/paper/sections_v5/appendix_v6.tex | head -300
  └ 301 lines...

● Read more of eval section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '300,600p' /var/folders/2k/n00k2r_d7sb3g3g0jkf059mh0000gn/T/copilo
  │ t-tool-output-1777530018587-b0vsql.txt
  └ 302 lines...

● Read more of eval section (post-freeze, Lean audit) (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '600,900p' /var/folders/2k/n00k2r_d7sb3g3g0jkf059mh0000gn/T/copilo
  │ t-tool-output-1777530018587-b0vsql.txt
  └ 302 lines...

● Check lean and test directories (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/ && ls tests/
  └ 42 lines...

● Read more eval - Lean audit section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '900,1100p' /var/folders/2k/n00k2r_d7sb3g3g0jkf059mh0000gn/T/copil
  │ ot-tool-output-1777530018587-b0vsql.txt
  └ 202 lines...

● Read handler soundness table (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat docs/paper/sections_v5/handler_soundness_table.tex
  └ 97 lines...

✗ Find mutation analysis in eval section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "mutation\|kill rate\|mutant\|7/50\|50 mut"
  │ docs/paper/sections_v5/eval_v6.tex | head -40
  └ Command not executed. The 'kill' command must specify at least one numeric
    PID. Usage: kill <PID> or kill -9 <PID>

● Find mutation analysis references (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -r "mutation" docs/paper/sections_v5/ | grep -v ".tex~" | head -40
  └ 20 lines...

● Find mutation analysis details (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "mutation\|7.50\|7\/50\|kill" docs/paper/sections_v5/eval_v6.tex
  │ | head -60
  └ 19 lines...

● Find Lean audit details in eval (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "Lean\|sorry\|lake\|11.11\|28.28\|28,000"
  │ docs/paper/sections_v5/eval_v6.tex | head -40
  └ 17 lines...

● Read mutation analysis section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '1180,1280p' /var/folders/2k/n00k2r_d7sb3g3g0jkf059mh0000gn/T/copi
  │ lot-tool-output-1777530018587-b0vsql.txt
  └ 102 lines...

● Read Lean audit section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '1279,1380p' /var/folders/2k/n00k2r_d7sb3g3g0jkf059mh0000gn/T/copi
  │ lot-tool-output-1777530018587-b0vsql.txt
  └ 103 lines...

Now I have enough to write the review. Let me check a few remaining claims in the codebase.

● Check real benchmarks artifacts (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls real_benchmarks/ && cat real_benchmarks/*.json 2>/dev/null | head -50
  │ || ls real_benchmarks/
  └ 52 lines...

● Check groundings.json structure (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python -c "import json; data=json.load(open('groundings.json'));
  │ print(list(data.keys())[:10])" 2>/dev/null || head -5 groundings.json
  └ 6 lines...

● Check claims in groundings.json (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python -c "
  │ import json
  │ with open('groundings.json') as f:
  │ data = json.load(f)…
  └ 6 lines...

● Check experimental artifacts (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls experiments_v5/ | head -20 && ls reproducibility/ | head -20
  └ 41 lines...

● Check soundness scope and block corpus reconciliation (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python -c "
  │ import json
  │ with open('reproducibility/bug_corpus_in_soundness_scope.json') as f:
  │ d = json.load(f)…
  └ 4 lines...

● Check block corpus reconciliation (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ /reproducibility && python -c "
  │ import json
  │ with open('block_corpus_488_reconciliation.json') as f:
  │ d = json.load(f)…
  └ 4 lines...

● Read reconciliation markdown (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/r
  │ eproducibility && head -80 block_corpus_488_reconciliation.md 2>/dev/null
  └ 78 lines...

● Check soundness scope details (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ /reproducibility && head -80 bug_corpus_in_soundness_scope.md 2>/dev/null
  │ || python3 -c "
  │ import json
  │ d = json.load(open('bug_corpus_in_soundness_scope.json'))…
  └ 83 lines...

Now I have a thorough understanding of the paper and supporting evidence. I'll write the review.

## Summary
TensorGuard is a static refinement-type checker for PyTorch `nn.Module` forward methods that verifies tensor shapes and gradient flow without executing or instantiating the module. The system emits five verdict types under a Z3-backed shape calculus, with Lean 4 mechanising 28 of 79 operator-rule handlers (11/11 soundness lemmas sorry-free). On a 60-bug historical corpus it achieves 53/60 Refuted-Proof; on a fragment-fair N=34 head-to-head with Pytea, 32/34 vs. 25/34 (McNemar p=0.0156). On a pre-registered unfiltered N=15 post-freeze sample, TG catches 5/15 vs. FakeTensorMode 2/15 vs. Pytea 3/15, explicitly non-significant. A backward verifier catches grad-flag bugs with a disclosed 2/8=25% worst-case false-verified rate on renamed-attribute parameter-sharing patterns. The 0-RP gap on the 488-block free-symbolic corpus is explicitly acknowledged upfront.

## Prior weakness disposition

- [PARTIAL] The most important practical limitation remains severe: on the 488-block real-source corpus, the user-visible free-symbolic regime still produces 0 unconditional RP verdicts (Section 4.1), so the deployed natural-distribution bug-finding... -- Paper now quantifies the under-input-shape-contract rate (15/295 analysable blocks), characterises 3/12 named LW→RP candidates as measured-flipped, and provides a per-category breakdown of principled abstentions, but the free-symbolic 0-RP headline is unchanged.
- [UNRESOLVED] The only clearly unbiased generalization test is the pre-registered unfiltered post-freeze sample in Table 3, and its 5/15 vs. 2/15 vs. 3/15 outcome is explicitly non-significant (p=0.39 vs. FakeTensorMode, p=0.68 vs. Pytea), which l... -- Same p-values; Benjamini–Hochberg correction leaves all adjusted p=1.00; no second wave added; power calculation locates the minimum additional N to reach significance at N_new≈26–77.
- [PARTIAL] The formal-sounding Lean-audited message still overhangs a much narrower real-corpus footprint: Section 4.4 says only 36/185 in-soundness verdicts on the 488-block corpus touch only Lean-or-pen-paper audited handlers, while 105/185 touch... -- The in-soundness footprint improved to 62/185 (from 36/185), and the tested-only-touching count fell to 66/185 (from 105/185); 62+66=128≠185 however leaves 57/185 in-soundness verdicts uncharacterised in the scope table (see Weakness 2 below).
- [PARTIAL] The backward-verifier story is improved but still limited: the 10-model real-world sweep excludes torch.utils.checkpoint and explicit parameter-sharing regimes, while Section 6 still concedes silent misclassification under renamed-attr... -- Paper now reports a concrete worst-case false-verified rate of 2/8=25% on the renamed-attribute construct family and explicitly distinguishes regex-detectable prevalence (≤12%) from the semantic-alias rate; the gap between these two bounds remains unquantified for the broader population.
- [PARTIAL] Theorem 5 is carefully scoped, but its significance is still modest: it is only a necessary-direction statement, pinned to torch 2.9.1, and the empirical audit uses surrogate contracts for some transformer cases rather than fully end-to-... -- Paper now labels C4 "exploratory" consistently throughout, extends the CNN-only end-to-end audit to 14 modules (all 19 recompile events on TG catalogue variables), and is explicit that 13/17 modules in the original audit use surrogate contracts; the theorem's scope is now clearly delimited.
- [PARTIAL] The mutation analysis remains weaker than I would like for a paper emphasising soundness-facing guarantees: the reported union kill rate is 7/50, which suggests that the current evaluation does not stress much of the analyser's implement... -- Targeted per-handler analysis now achieves conv2d 20/38=53% and einsum 7/7=100% on comparison-flip/arithmetic-swap mutations using an extended targeted corpus; however the global union rate remains 7/50=14%, and the targeted result uses an incommensurable extended corpus.

## Strengths
- Exceptional pre-emptive self-calibration: the paper names every substantive limitation—0-RP on free-symbolic corpus, 2/8=25% false-verified on worst-case grad pattern, non-significant N=15—before any reviewer could flag them, and provides quantitative bounds throughout.
- The 7/7 natural HuggingFace upstream bug catch (across Llama, Qwen2, Mistral, Phi-3, cited PR/issue per row, no injected variants) is the paper's most compelling practical demonstration; it shows TG operates on real class source as found in the wild.
- The McNemar-exact head-to-head with Pytea (32/34 vs. 25/34, p=0.0156, paired-bootstrap CI lower bound +8.8 pp above zero) is statistically sound and properly corrected for catalogue confound via fragment-fairness enforcement at verification time.
- The Lean operator-rule audit is honestly scoped—the rule table only, not the implementation—and all 11 previously-axiomatic soundness lemmas are now sorry-free including the corrected `permList_compose_inrange` restatement.
- The reproducibility artefact is unusually thorough: SHA-pinned manifests, per-block verdict JSONs, the full Pytea matched-pair contingency table, the AST-extractor oracle cross-validation, the mutation harness per-handler breakdowns, and power calculations are all released.

## Weaknesses
- The N=15 post-freeze test is the only pre-registered unbiased generalisation test, and after Benjamini–Hochberg correction all three pairwise Fisher-exact p-values adjust to 1.00 (raw p=0.39, 0.68, 1.00). The paper's power calculation shows N_new≈26 (TG vs. FakeTensorMode, one-sided) and N_new≈77 (TG vs. Pytea, one-sided) to reach p<0.05. At N=15, "a directional trend, not a significance claim" is the honest characterisation, but it also means the paper's strongest unbiased test cannot distinguish TG from either baseline at any standard α.
- The handler-scope arithmetic is inconsistent: the abstract and Section 4.4 report 62/185 in-soundness verdicts touching only the Lean-or-pen-paper audited footprint, and 66/185 touching at least one tested-only handler, but 62+66=128≠185; the remaining 57/185 in-soundness verdicts (15 Verified, 42 CV) are not explicitly categorised. If these 57 use only pen-and-paper handlers, the 62 count should absorb them; if they use out-of-catalogue routes, the paper should say so.
- Table 1's caption states "all **56** refutations are Refuted-Proof" while the table body shows R=53 for TG on the bug corpus and the abstract/body consistently report "53/60 (88.3%)." Either the caption retains a stale number from an earlier version or the table count is wrong; the reproducibility artefact's soundness-scope report also uses "56 RP verdicts" for the same corpus, creating a three-way inconsistency (53 vs. 56 vs. Table-1 body).
- The 7/7 natural upstream HuggingFace bugs are presented without a torch.compile baseline. On the fragment-fair N=34 subset torch.compile achieves 34/34; since the 7 natural bugs are presumably importable (they come from public fix-PRs), the torch.compile catch rate on these 7 would clarify whether TG's advantage over execution-based tools persists specifically on naturally-occurring class source.
- The global mutation kill rate (7/50 union, 14%) remains the headline for analyser-wide robustness. The targeted per-handler improvement (conv2d 53%, einsum 100%) is reported on an extended 18-case targeted corpus constructed specifically to cover those handlers, making it not directly comparable to the 7/50 figure. A unified measurement—the targeted extension corpus plugged into the same 50-mutant sweep—would give a single comparable rate.

## Questions
- The handler-scope table reports 62+66=128/185 in-soundness verdicts explicitly categorised; what are the remaining 57/185? Are they Verified/CV verdicts that use only pen-and-paper handlers but no Lean-audited and no tested-only handler—and if so, why does "Lean-or-pen-paper audited footprint" not absorb them into the 62?
- Table 1 caption says "all 56 refutations are Refuted-Proof"; the table, abstract, and body all say 53. Which number is correct? The soundness-scope artefact also says "56 RP verdicts" for the 60-bug corpus.
- Can you report torch.compile's catch rate on the 7 natural HuggingFace upstream bugs? These are real fix-PR sources and are presumably importable, making this the fairest apples-to-apples comparison between TG and the strongest execution-based baseline on naturally-occurring defects.
- The real-corpus ablation (Section 4.3) shows that all five feature knobs are flat lines on the upstream-faithful 10-bug corpus. Since CEGAR and phase-check are confirmed non-functional on all tested corpora, and the three discriminative knobs (device-consistency, gradient-flow, low-confidence) each contribute only on the hand-designed stress set: is there any real-corpus evidence that any single feature beyond the base fragment is load-bearing for the reported headlines?
- The power calculation places the minimum additional N to separate TG from FakeTensorMode at N_new≈26 (one-sided). Given that the pre-registered query is already defined and frozen, is there a plan to collect this second wave? A null answer is acceptable, but the current N=15 result provides essentially no evidence of practical superiority over the cheapest available baseline on an unbiased distribution.

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons
Providing a statistically significant improvement over execution-based baselines on a pre-registered unfiltered post-freeze corpus—at minimum a second wave of N≈26 against FakeTensorMode, confirming the 5/15 vs. 2/15 direction—would push this to a 7 by establishing that TG's advantage on natural-distribution bugs is real rather than a point-estimate artefact of an underpowered N=15 sample.


Changes   +0 -0
Requests  1 Premium (5m 39s)
Tokens    ↑ 923.5k • ↓ 16.1k • 836.0k (cached)
