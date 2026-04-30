● Read round17_reviewer.md
  │ .comet_neurips/round17_reviewer.md
  └ 121 lines read

● Check git status (shell)
  │ git --no-pager status --short
  └ 277 lines...

I’m pulling the paper and the concrete result artifacts together first so the review tracks what’s actually in the current repo, not stale earlier claims. Then I’ll score only the current version against the NeurIPS bar.

● Explore(claude-sonnet-4.6) Audit prior weaknesses
  └ Agent started in background with agent_id: prior-weakness-audit. You'll ...

● Read neurips.txt
  │ docs/paper/neurips.txt
  └ L1:240 (240 lines read)

● Read neurips.txt
  └ L1:240 (240 lines read)

The current paper text already narrows several claims and demotes some earlier overstatements, so I’m checking whether the evidence actually matches that tightened framing. I’m also pulling the exact numbers behind the remaining theoretical and empirical weak spots before I score it.

● Read neurips.txt
  │ docs/paper/neurips.txt
  └ L520:760 (241 lines read)

● Read neurips.txt
  │ docs/paper/neurips.txt
  └ L760:1040 (281 lines read)

○ Search (grep)
  │ "5/15|53/60|32/34|25/34|11/57|25/128|103/185|28,000/28,000|500/500|8/8|0/50|
  │ 14-module|Theorem
  │ 5|McNemar|Fisher|post-freeze|parameter-sharing|silent|mutation|kill
  │ rate|conv2d|14-module|surrogate" (docs/paper/neurips.txt, neurips.txt,
  │ experiments, experiments_v5, benchmarks)
  └ Timed out

● Read neurips.txt
  │ docs/paper/neurips.txt
  └ L1040:1360 (321 lines read)

● Read neurips.txt
  │ docs/paper/neurips.txt
  └ L1360:1700 (341 lines read)

● Search (grep)
  │ "post-freeze|Fisher|5/15|2/15|3/15|grad-flag|parameter-sharing|checkpoint|si
  │ lent verified|500/500|10-module|real-world sweep|mutation|kill|Theorem
  │ 5|14-module|surrogate|11/57|25/128|tested-only|soundness
  │ scope|28,000/28,000" (docs/paper/neurips.txt)
  └ 7 lines found

I’ve got the updated empirical story: the authors now explicitly report several formerly headline-negative results as limitations rather than wins. The remaining question is whether the residual evidence base is strong enough for acceptance once those caveats are properly discounted.

● Read neurips.txt
  │ docs/paper/neurips.txt
  └ L1688:1960 (273 lines read)

● Read neurips.txt
  │ docs/paper/neurips.txt
  └ L1960:2260 (301 lines read)

● Search (grep)
  │ "Section 6|6\.|parameter-sharing|checkpoint|grad-flow|silent verified|silent
  │ incorrect|prevalence|12%|mutant|exposure|backward verifier|real-world
  │ sweep|resnet18|vit_b_16|bert-base|gpt2|500/500" (docs/paper/neurips.txt)
  └ 71 lines found

● Search (glob)
  │ "**/{run_dynamo_correspondence_v5.py,grad_flag_torchvision_holdout.py,verify
  │ _real_bugs_postfreeze.py,verify_real_bugs_unfiltered.py,track_C_coverage.jso
  │ n,lean_parity_v5_results.json,v5_baseline_comparison.json,dynamo_corresponde
  │ nce_v5.json}"
  └ 8 files found

● Search (glob)
  │ "**/{test_lean_parity.py,test_lean_handler_parity.py,test_refinement_track_c
  │ .py,test_feature_stress.py}"
  └ 4 files found

● Check key artifact files (shell)
  │ python - <<'PY'
  │ import json, pathlib
  │ paths = [
  │ 'benchmarks/headline_results.json',
  │ 'experiments_v5/v5_baseline_comparison.json',…
  └ 22 lines...

The repo does contain the new evaluation and audit artifacts the paper points to, so I’m checking whether they line up with the paper’s tightened claims rather than just existing by name. The remaining gap is mainly judgment, not missing evidence.

● Read dynamo_correspondence_v5.json
  │ experiments_v5/dynamo_correspondence_v5.json
  └ L1:220 (220 lines read)

● Read v5_baseline_comparison.json
  │ experiments_v5/v5_baseline_comparison.json
  └ L1:200 (200 lines read)

● Summarize key JSON artifacts (shell)
  │ python3 - <<'PY'
  │ import json, pathlib
  │ paths = [
  │ 'benchmarks/headline_results.json',
  │ 'experiments_v5/v5_baseline_comparison.json',…
  └ 31 lines...

Now I have enough information to give a complete assessment. Let me check one more thing about the abstract's specific claim on the 7 naturally-occurring bugs vs. the prior criticism:I now have comprehensive information to deliver the review assessment.

---

## Assessment of Prior Weaknesses — Current Paper/Repo

---

### 1. Natural-distribution bug-finding result is essentially negative (0 unconditional RP on 488-block corpus)

**Status: PARTIAL**

The 0-RP ceiling on the 488-block free-symbolic-config regime is preserved verbatim in the abstract and §Eval (lines 67–81 of `eval_v6.tex`), explicitly framing it as "the unconditional-RP claim is carried by the bug corpora, not by the block corpus." Partially resolved because a *new* 7/7 naturally-occurring RP result on real HuggingFace transformers PRs (Llama, Qwen2, Mistral, Phi-3) is added as genuine natural-distribution evidence — but that set (N=7) was manually transcribed/extracted, not mined from the same 488-block pool, so the original negative result on real, unfiltered library source stands intact.

---

### 2. Fragment-fair head-to-head against a frozen 2022 Pytea baseline; no torch.compile(fullgraph=True) comparison

**Status: PARTIAL**

The paper now explicitly confirms Pytea has zero commits after `cb02a8a` (2022-04-26), providing justification for using the frozen baseline (`eval_v6.tex` §"Pytea modern-subset filter"). `torch.compile` (PyTorch 2.9.1) with `FakeTensor`/`fullgraph=True` *is* now run on the same 34-bug fragment-fair subset — and it scores **34/34**, *exceeding* TG's 32/34; this is disclosed honestly. The structural asymmetry (torch.compile inapplicable to 481/488 real-source blocks) is the stated differentiator, but on the fragment-fair head-to-head the contemporary baseline now outperforms TG, which is a live weakness not a resolution.

---

### 3. Pre-registered post-freeze evaluation fails to separate from baselines (Fisher exact not significant)

**Status: PARTIAL**

The paper now explicitly reports two-sided Fisher exact p=0.39 (TG vs. FakeTensorMode) and p=0.68 (TG vs. Pytea) on N=15, applies Benjamini–Hochberg correction (every BH-adjusted p=1.00), and clearly states "we do not rely on a Bayesian supplement to upgrade the claim" (`eval_v6.tex` lines 646–692). It has been reframed from a significance claim to "a directional trend." The underlying data (5/15 vs 2/15 vs 3/15) and statistical non-separation are unchanged; the paper is transparent about the weakness but has not resolved it empirically.

---

### 4. Soundness coverage much narrower than headline suggests (only 36/185 in-soundness verdicts touch fully audited handlers)

**Status: UNRESOLVED**

The figure is preserved exactly: "36/185 in-soundness verdicts in total" touch only the Lean-or-pen-paper-audited footprint (`eval_v6.tex` line 1350), with 103/185 (55.7%) touching at least one of the 44 tested-only handlers. The abstract and §Lean-audited operator-rule table both disclose this prominently. No new Lean lemmas were added to audited handlers to reduce the 44-tested-only gap; 7 pen-and-paper handlers have been added (flatten/squeeze/unsqueeze/softmax as T-Identity), shifting the abstract's count from 28 to "28+7=35 in-soundness footprint," but the 36/185 overall figure and the 44 tested-only handlers remain.

---

### 5. Grad-flag claim has 25% worst-case runtime false-verified rate on parameter-sharing/checkpoint family

**Status: UNRESOLVED (but now fully disclosed)**

The `limconc_v6.tex` text explicitly retains and highlights the **2/8 = 25.0% false-verified rate** on the held-out worst-case construct family (tied/renamed-attribute parameter sharing, gradient checkpointing). New supplementary evidence (0/16 Track-E modules affected, 0/2908 renamed-attribute hits in transformers+timm AST sweep, 6/6 correctly True-Verified on tied BERT/GPT2/T5/BART/RoBERTa+minimal repro) is added but does not fix the core 2/8 corner case. The paper correctly labels the 6/6 result as "no silently-incorrect verdict" on models where the tied parameter genuinely does receive a gradient — but the 2/8 renamed-attribute pattern that evades detection is explicitly called a known limitation, not resolved.

---

### 6. 500/500 backward-verifier agreement on grammar-generated tiny modules only; 10-module real-world sweep inadequate

**Status: PARTIAL**

The paper now adds a 10-module real-world sweep (ResNet18, ViT-b-16, BERT, GPT-2, DistilBERT, T5-small, Whisper-tiny, CLIP, Wav2Vec2) with 10/10 predicted `requires_grad` topologies matching runtime `loss.backward()` ground truth (`eval_v6.tex` lines 1400–1412). However, the paper simultaneously and explicitly notes "None of these models exercise `torch.utils.checkpoint` or explicit parameter sharing" — the exact failing regimes from weakness 5. The 6/6 tied-weight positive-control result is also added. The weakness is partially addressed: real models now tested, but the distribution gap (all 10 are standard unshared-parameter models) is openly acknowledged.

---

### 7. Mutation-kill rates weak for a soundness-oriented paper (7/50 = 14% triple-corpus union)

**Status: PARTIAL**

The 7/50 = 14% multi-corpus AST union rate is explicitly preserved. The paper adds a *targeted* kill-rate analysis on the four load-bearing handlers (`view`/`reshape`, broadcasting, `conv2d`, `einsum`): with a 18-case extension corpus and exhaustive per-line enumeration of comparison-flip and arithmetic-swap mutants, it reports `conv2d` 20/38 = 53%, `einsum` 7/7 = 100%, union 27/45 = 60% (`eval_v6.tex` lines 1224–1272). The headline global rate (7/50) remains weak and is acknowledged; the targeted handler rates substantially improve the picture for the handlers most central to the headline results, but they depend on a purpose-built extension corpus rather than the natural corpora.

---

### 8. Theorem 5 (Dynamo-guard correspondence) carries little weight; necessary-direction only; 16/17 modules use forward-signature surrogate

**Status: PARTIAL**

The paper now: (a) explicitly re-titles it as "necessary direction" throughout; (b) confirms 16/17 modules use the "documented forward-signature surrogate" because full instantiation exceeds end-to-end constraint solving; (c) adds an extended 14-module end-to-end audit (9 CNN fully end-to-end, 4 transformer via surrogate, 1 ResNet50 layer) with 0 falsifying instances on 19 recompile events; (d) provides an explicit falsification predicate; and (e) pins the freeze SHA. The 4 transformer blocks still use the surrogate workaround, the theorem remains one-directional, and it is still audited against frozen torch 2.9.1. The paper's framing is now appropriately calibrated ("treat as calibration of Dynamo against hand-written contracts rather than as an in-theorem instantiation"), but the structural limitation is unresolved.

---

## Strongest Remaining Weaknesses for Current Round

1. **torch.compile beats TG on its own ground**: On the fragment-fair 34-bug subset, `torch.compile`/FakeTensor scores 34/34 vs TG's 32/34 — the key contemporary baseline *outperforms* the system when it is applicable. The paper's core differentiation rests entirely on the 481/488 inapplicability observation, which is structural but narrows the practical novelty claim significantly.

2. **Pre-registered post-freeze evaluation not statistically separable**: The only uncontaminated, pre-registered evaluation (N=15 unfiltered PRs) yields p=0.39/0.68 with BH-corrected p=1.00 for all pairwise contrasts. The power calculation projects N≈77–N≈100 new bugs needed for significance vs. Pytea — roughly 5× the current sample. This is an open empirical gap at the core of the experimental design.

3. **Soundness coverage gap (36/185 = 19%)**: The vast majority of in-soundness verdicts on real library source traverse at least one tested-only handler, so the Lean mechanization provides formal backing for only ~19% of the live verdict surface. The gap between the "Lean-audited" branding and the actual coverage is significant.

4. **25% false-verified rate on worst-case grad-flag construct family (2/8)**: Explicitly disclosed but unresolved; the regex-prevalence argument (≤12% of training scripts) is incomplete because it relies on a narrow literal-pattern regex, and the paper acknowledges semantic-alias patterns evade it.

5. **Global mutation-kill rate 7/50 = 14%**: The targeted handler rates are strong (60% union, 100% einsum), but they required a purpose-built extension corpus; the natural-distribution multi-corpus rate remains at 14%, which is low for a soundness-claiming system.

6. **N=7 for the new "naturally-occurring bugs" claim**: The headline positive result now added to the abstract (7/7 real HuggingFace PR bugs) is based on manually transcribed/minimally adapted modules from 7 PRs across 4 decoder families — a small, potentially curated set that is not independently pre-registered.

---

## Overall Accept/Reject Leaning

**Leaning: Weak Reject / Borderline**

The paper has substantially improved its epistemic honesty — almost every prior weakness is now explicitly disclosed with quantified bounds. The theoretical contribution (refinement-type calculus, assume/guarantee discipline, Lean-mechanized rule table) is sound and non-trivial. However, the empirical case for practical superiority remains unconvincing: the contemporary baseline (torch.compile) outperforms TG on the only clean head-to-head where all tools apply; the pre-registered post-freeze evaluation is underpowered and non-significant; the Lean coverage backs only 19% of live verdicts; and the grad-flag false-verified rate at 25% on a corner case that the paper itself frames as a known construct family is unresolved. The paper is honest about all of this, which is commendable, but honesty about limitations does not substitute for empirical strength. The gap between the formal claims and the deployed-system evidence is too large for an accept at a top-tier venue without a stronger positive empirical result or a substantially enlarged pre-registered evaluation.● Read (Explore agent — Audit prior weaknesses)
  └ Completed

## Summary
This paper presents TensorGuard, a static no-execution verifier for PyTorch `nn.Module` code that reasons about tensor shapes and gradient-flow properties from class source rather than from instantiation or tracing. The core technical contribution is a refinement-type calculus with assume/guarantee contracts at module boundaries, plus a five-way verdict taxonomy that separates sound refutations from conservative warnings and abstentions. The paper also contributes a partial Lean 4 audit of the operator-rule table and a backward verifier for several canonical silent-zero-grad bug patterns. Empirically, the paper reports strong results on a curated 60-bug historical corpus and improved upstream-faithful real-bug reproductions, while being much more careful than earlier versions about stating that the 488-block real-source corpus yields 0 unconditional RP verdicts in the user-visible free-symbolic regime. It also presents a small pre-registered post-freeze evaluation and a necessary-direction correspondence result relating TensorGuard refinements to TorchDynamo guards.

## Prior weakness disposition
- [PARTIAL] **The natural-distribution bug-finding result is essentially negative and the paper acknowledges it.** On the 488-block real-source corpus... -- The current paper now states this limitation prominently and treats the 0-RP real-source result as fragment coverage rather than bug-finding, but the underlying negative result remains.
- [PARTIAL] **The fragment-fair head-to-head is against a frozen 2022 baseline.** Pytea's last upstream commit is `cb02a8a` (2022-04-26) per... -- The paper now documents that Pytea is in fact frozen at that commit and tightens the framing, but the clean fragment-fair head-to-head is still mainly against that old baseline.
- [PARTIAL] **The pre-registered post-freeze evaluation, which is the only attempt at unbiased generalisation, fails to separate from baselines.** ... -- The paper now reports the non-significant Fisher tests explicitly and stops overselling the result, but 5/15 vs. 2/15 vs. 3/15 is still too small to establish a clear win.
- [UNRESOLVED] **Soundness coverage on the deployed system is much narrower than the headline suggests.** Of the 185 in-soundness verdicts on... -- The paper still reports only 36/185 real-corpus in-soundness verdicts as lying wholly inside the Lean-or-pen-paper audited footprint.
- [PARTIAL] **The grad-flag claim has a 25% worst-case runtime false-verified rate on the construct family that matters.** The held-out runtime... -- The paper now scopes and audits this limitation much more carefully, but the first-order grad lattice still excludes the problematic parameter-sharing/checkpointing regime rather than solving it.
- [PARTIAL] **The `500/500` static↔runtime backward-verifier agreement is on grammar-generated tiny modules**, not on a meaningful distribution... -- The paper adds a 10-model real-world sweep, which is useful, but those models do not exercise the hardest failure modes emphasized in the limitation section.
- [PARTIAL] **Mutation-kill rates are weak for a soundness-oriented paper.** The triple-corpus union kill rate is 7/50 = 14%; even after... -- The paper adds more nuance and targeted analysis, but the global union kill rate reported in the main text is still low for a system making strong soundness-facing claims.
- [PARTIAL] **Theorem 5 (Dynamo-guard correspondence) carries little theoretical weight.** It is a necessary-direction inclusion proved against... -- The paper now scopes this theorem much more honestly, but it remains a one-way correspondence audited on a small frozen-snapshot study with surrogate contracts for some transformer cases.

## Strengths
- The paper is substantially more calibrated than many systems papers in this area: it distinguishes RP/CV/LW/Abstain carefully and is unusually explicit about what is and is not covered by the theorem.
- The refinement-type formulation with assume/guarantee composition at the `nn.Module` boundary is technically interesting and more principled than a purely engineering-driven shape checker.
- The paper targets a practically important niche where execution-based tools are often structurally inapplicable because real modules are hard to instantiate or trace.
- The Lean audit is meaningful as a rule-table audit: 28 handlers mechanized, 11/11 soundness lemmas closed sorry-free, and parity checks against PyTorch are helpful evidence even if they do not cover the whole analyzer.

## Weaknesses
- The most important practical limitation remains severe: on the 488-block real-source corpus, the user-visible free-symbolic regime still produces 0 unconditional RP verdicts (Section 4.1), so the deployed natural-distribution bug-finding story is weak.
- The only clearly unbiased generalization test is the pre-registered unfiltered post-freeze sample in Table 3, and its 5/15 vs. 2/15 vs. 3/15 outcome is explicitly non-significant (`p=0.39` vs. FakeTensorMode, `p=0.68` vs. Pytea), which leaves the empirical advantage uncertain.
- The formal-sounding Lean-audited message still overhangs a much narrower real-corpus footprint: Section 4.4 says only 36/185 in-soundness verdicts on the 488-block corpus touch only Lean-or-pen-paper audited handlers, while 105/185 touch at least one tested-only handler.
- The backward-verifier story is improved but still limited: the 10-model real-world sweep excludes `torch.utils.checkpoint` and explicit parameter-sharing regimes, while Section 6 still concedes silent misclassification under renamed-attribute parameter sharing.
- Theorem 5 is carefully scoped, but its significance is still modest: it is only a necessary-direction statement, pinned to torch 2.9.1, and the empirical audit uses surrogate contracts for some transformer cases rather than fully end-to-end verification.
- The mutation analysis remains weaker than I would like for a paper emphasizing soundness-facing guarantees: the reported union kill rate is 7/50, which suggests that the current evaluation does not stress much of the analyzer’s implementation surface.

## Questions
- The paper is right to emphasize structural inapplicability of execution-based baselines on the 488-block corpus, but why is there not a larger fragment-fair comparison against a contemporary execution-based baseline on the subset where such tools do apply?
- For the 36/185 audited-footprint number, can the authors break down which tested-only handlers account for most real-corpus verdicts and whether those same handlers dominate the post-freeze catches?
- In Section 6, how robust is the prevalence estimate for renamed-attribute parameter sharing beyond the literal AST patterns used in the sweep?
- Given the Table 3 power discussion, what is the smallest additional post-freeze sample the authors believe is realistically needed to turn the current directional trend into persuasive evidence?

## Scores
Soundness: 2
Presentation: 4
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
A substantially larger pre-registered post-freeze evaluation that actually separates from baselines, or a much larger audited real-verdict footprint than 36/185, would likely move my score up by one point. As written, the paper is thoughtful and technically interesting, but the strongest empirical and formal claims still stop short of the NeurIPS accept bar.


Changes   +0 -0
Requests  1 Premium (3m 53s)
Tokens    ↑ 744.9k • ↓ 13.9k • 639.2k (cached) • 2.0k (reasoning)
