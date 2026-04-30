● Read round08_reviewer.md
  │ .comet_neurips/round08_reviewer.md
  └ 119 lines read

I’m pulling the paper claims and the repo evidence side by side so the review can be anchored to what’s actually substantiated.

● Read README.md
  └ L1:260 (257 lines read)

● Read API.md
  └ L1:220 (219 lines read)

● Search (grep)
  │ "mutation|conv2d|einsum|488|185|hybrid|complementarity|Theorem
  │ 5|Dynamo|recompile|grad-flag|silent-error|regex-detectable|post-freeze|unfil
  │ tered|FakeTensor|Pytea|Theorem 5|SHAPE" (neurips.txt)
  └ 7 lines found

● Search (grep)
  │ "32/34|22/34|15/15|2/15|3/15|185|488|36|105/185|hybrid|TG-only|FT-only|20\)|
  │ 20,|zero overlap|13 SHAPE|10 CNN|4
  │ transformer|2/8|25%|12%|0/10|conv2d|einsum" (neurips.txt)
  └ 8 lines found

● Search (glob)
  │ "experiments_v5/**/*{conv2d,einsum,hybrid,postfreeze,real,holdout,power,muta
  │ tion}*"
  └ 22 files found

● Search (grep)
  │ "conv2d|einsum|mutation|kill|hybrid|postfreeze|power analysis|grad|runtime
  │ holdout|Dynamo|recompile|transformer" in *.{py,md,txt,tex,json,csv,log}
  │ (experiments_v5, reproducibility, docs)
  └ 9 files found

I found the main artifacts. Now I’m checking whether the paper’s headline claims are backed by the released experiments, especially the six previously flagged weak points.

● Read neurips.txt
  └ L1:80 (80 lines read)

● Read neurips.txt
  └ L170:280 (110 lines read)

● Read neurips.txt
  └ L80:170 (91 lines read)

● Read neurips.txt
  └ L280:420 (141 lines read)

● Search (grep)
  │ "Table 4|Table 5|Section 4|Section 3.2|Section
  │ 4.1|4\.1|4\.2|4\.3|4\.4|Theorem 5|Dynamo|hybrid|post-freeze|real-public|sile
  │ nt-zero|grad-flag|load-bearing|mutation" (neurips.txt)
  └ 8 lines found

● Search (grep)
  │ "def .*mutation|mutation|conv2d|einsum|grad_flag|Theorem
  │ 5|Dynamo|recompile|FakeTensor|Pytea|hybrid" in *.py (tests, experiments_v5,
  │ reproducibility)
  └ 7 lines found

I’m narrowing to the few artifacts that directly answer the prior-review checklist: post-freeze significance, hybrid complementarity, mutation stress, grad runtime holdout, and Theorem 5’s empirical audit.

● Read postfreeze_power_analysis.py
  │ reproducibility/postfreeze_power_analysis.py
  └ L1:260 (224 lines read)

● Read run_hybrid_mode.py
  │ experiments_v5/run_hybrid_mode.py
  └ L1:260 (160 lines read)

● Read grad_flag_torchvision_holdout.py
  │ experiments_v5/grad_flag_torchvision_holdout.py
  └ L1:260 (209 lines read)

● Read dynamo_theorem5_n200.md
  │ reproducibility/dynamo_theorem5_n200.md
  └ L1:260 (170 lines read)

● Read hybrid_mode_results.json
  │ experiments_v5/hybrid_mode_results.json
  └ L1:220 (220 lines read)

● Search (glob)
  │ "**/*mutation*"
  └ 29 files found

● Read mutation_kill_rate_loadbearing_v2.json
  │ reproducibility/mutation_kill_rate_loadbearing_v2.json
  └ L1:240 (240 lines read)

● Read mutation_kill_rate_loadbearing_v2.md
  │ reproducibility/mutation_kill_rate_loadbearing_v2.md
  └ L1:240 (80 lines read)

● Search (grep)
  │ "Theorem 5|Dynamo|14-module|nine CNN|four transformer|preliminary|surrogate"
  │ (neurips.txt)
  └ 7 lines found

● Search (grep)
  │ "2/8|25%|12%|parameter-sharing|renamed-attribute|grad-flag|silent
  │ incorrect|silent-error|worst-case" in *.{py,md,json,txt,tex} (neurips.txt,
  │ reproducibility, experiments_v5)
  └ 7 lines found

● Search (grep)
  │ "185|488|36|105/185|44 tested-only|tested-only
  │ handlers|Lean|pen-paper|audited footprint|user-visible|free-symbolic|0
  │ unconditional|in-soundness|soundness" in *.{py,md,json,txt,tex,jsonl}
  │ (neurips.txt, reproducibility, experiments_v5)
  └ 9 lines found

● Read review_response.md
  └ L1:260 (260 lines read)

● Search (grep)
  │ "^3\.2|^4\.1|^4\.2|^4\.3|^4\.4|^5 |^6 |^7
  │ |Historical|Real-public|post-freeze|hybrid|stress|Table 4|Table 5|Table
  │ 7|Theorem 5" (neurips.txt)
  └ 70 lines found

● Read neurips.txt
  └ L1048:1115 (68 lines read)

● Read neurips.txt
  └ L1290:1405 (116 lines read)

● Read neurips.txt
  └ L1208:1256 (49 lines read)

● Read neurips.txt
  └ L1578:1605 (28 lines read)

● Read neurips.txt
  └ L3388:3405 (18 lines read)

● Read neurips.txt
  └ L1768:1800 (33 lines read)

● Read neurips.txt
  └ L1410:1565 (156 lines read)

✗ Read grad_flag_torchvision_holdout.json
  │ reproducibility/grad_flag_torchvision_holdout.json
  └ Path does not exist

● Read neurips.txt
  └ L1565:1595 (31 lines read)

● Search (glob)
  │ "experiments_v5/v8/*user_visible*"
  └ 5 files found

● Search (glob)
  │ "reproducibility/*handler_scope*"
  └ 6 files found

I’ve got enough to see some prior issues were addressed, but a few of the new fixes still look fragile or stress-set-only. I’m checking the exact scope of the soundness footprint and whether the released artifacts actually support the new wording.

● Read handler_scope_per_block.md
  │ reproducibility/handler_scope_per_block.md
  └ L1:220 (46 lines read)

● Read postfreeze_5catches_handler_scope.md
  │ reproducibility/postfreeze_5catches_handler_scope.md
  └ L1:200 (20 lines read)

● Read user_visible_rp.json
  │ experiments_v5/v8/user_visible_rp.json
  └ L1:220 (26 lines read)

● Read grad_lattice_runtime_holdout.py
  │ reproducibility/grad_lattice_runtime_holdout.py
  └ L1:260 (260 lines read)

● Search (grep)
  │ "postfreeze_power_analysis\.md|dynamo_theorem5_n200\.md|mutation_kill_rate_l
  │ oadbearing_v2\.md|handler_scope_per_block\.md|user_visible_rp\.json|grad_lat
  │ tice_runtime_holdout" (README.md, neurips.txt)
  └ 3 lines found

● Read postfreeze_power_analysis.md
  │ reproducibility/postfreeze_power_analysis.md
  └ L1:120 (21 lines read)

● Read mutation_kill_rate_loadbearing.md
  │ reproducibility/mutation_kill_rate_loadbearing.md
  └ L1:180 (42 lines read)

● Read mutation_kill_rate_corpora.md
  │ reproducibility/mutation_kill_rate_corpora.md
  └ L1:180 (51 lines read)

● Read grad_lattice_runtime_holdout.py
  │ reproducibility/grad_lattice_runtime_holdout.py
  └ L260:520 (239 lines read)

● Search (glob)
  │ "reproducibility/grad_lattice_runtime_holdout.*"
  └ 5 files found

● Read grad_lattice_runtime_holdout.json
  │ reproducibility/grad_lattice_runtime_holdout.json
  └ L1:220 (157 lines read)

● Read grad_lattice_runtime_holdout.md
  │ reproducibility/grad_lattice_runtime_holdout.md
  └ L1:180 (42 lines read)

● Read grad_lattice_runtime_holdout.log
  │ reproducibility/grad_lattice_runtime_holdout.log
  └ L1:200 (15 lines read)

● Read neurips.txt
  └ L463:548 (86 lines read)

● Read real_bugs_unfiltered.md
  │ reproducibility/real_bugs_unfiltered.md
  └ L1:220 (110 lines read)

● Search (glob)
  │ "reproducibility/*real_bugs_unfiltered*"
  └ 2 files found

● Search (grep)
  │ "5/15|0\.39|0\.68|5/15 catches|2/15|3/15" in *.{md,json,py}
  │ (reproducibility, experiments_v5)
  └ 9 lines found

● Search (grep)
  │ "42%|73%|53%|100%|21/50|8/11|20/38|7/7|0/10" (neurips.txt)
  └ 11 lines found

● Search (grep)
  │ "2/8|25%|6/8|0/8|parameter sharing|tied|renamed-attribute|0.17%|11.45%"
  │ (neurips.txt)
  └ 53 lines found

● Search (grep)
  │ "36/185|105/185|11/57|25/128|72/128|33/57|8/34|20/34" (neurips.txt)
  └ 4 lines found

● Read mutation_kill_rate_loadbearing_v2.py
  │ reproducibility/mutation_kill_rate_loadbearing_v2.py
  └ L1:260 (260 lines read)

● Search (grep)
  │ "load-bearing|comparison-flip|arithmetic-swap|conv_channel_mismatch|einsum_d
  │ im|42%|73%|53%|100%|29/61|27/45" in *.{txt,md,py,json} (neurips.txt,
  │ reproducibility)
  └ 7 lines found

● Read neurips.txt
  └ L1198:1210 (13 lines read)

● Search (grep)
  │ "57, 206, 225|57 206 225|225 TG abstains|ctor failed|exec failed"
  │ (neurips.txt)
  └ 4 lines found

● Search (grep)
  │ "false-verified rate|0/8|2/8|6/8|held-out runtime|trainer-realistic" in
  │ *.{md,json,txt,py,log} (reproducibility, neurips.txt)
  └ 93 lines found

● Read neurips.txt
  └ L1678:1692 (15 lines read)

● Read grad_lattice_hf_trainer_holdout.md
  │ reproducibility/grad_lattice_hf_trainer_holdout.md
  └ L1:120 (37 lines read)

● Read backward_param_sharing_audit.md
  │ reproducibility/backward_param_sharing_audit.md
  └ L1:160 (39 lines read)

I found a substantive mismatch: the paper still reports a `0/8` grad-lattice false-verified rate, but the shipped rewritten holdout artifact now reports `2/8`. I’m checking whether that inconsistency is isolated or part of a broader pattern in the reproducibility section.

● Read dynamo_theorem5_n200.py
  │ reproducibility/dynamo_theorem5_n200.py
  └ L1:220 (220 lines read)

● Search (grep)
  │ "107 candidate|55 completed|146|67|0/8|2/8|80|187|26|77|56" in
  │ *.{txt,md,json,py,log} (neurips.txt, reproducibility)
  └ 9 lines found

● Search (glob)
  │ "reproducibility/dynamo_theorem5_n200*"
  └ 4 files found

● Read dynamo_theorem5_n200.log
  │ reproducibility/dynamo_theorem5_n200.log
  └ L1:120 (120 lines read)

● Read dynamo_theorem5_n200.json
  │ reproducibility/dynamo_theorem5_n200.json
  └ L1:120 (120 lines read)

● Search (grep)
  │ "107 candidate|55 successful|146|67 successful|72 in-contract|0/8 =
  │ 0.0%|2/8|80 \| 187|26 for TG vs" (neurips.txt,
  │ reproducibility/dynamo_theorem5_n200.md,
  │ reproducibility/grad_lattice_runtime_holdout.md,
  │ reproducibility/postfreeze_power_analysis.md)
  └ 33 lines found

## Summary
This paper presents TensorGuard, a no-execution static verifier for PyTorch `nn.Module` source that uses refinement types and Z3 to reason about tensor shapes and a coarse gradient-flow lattice without instantiating the model. The formal core is a refinement-typed calculus with assume/guarantee module contracts, plus a Lean-audited operator-rule table covering 28 shape handlers and sorry-free proofs for 11 previously axiomatic lemmas. Empirically, the paper reports 53/60 proof-grade refutations on a historical bug corpus, a 32/34 vs. 22/34 head-to-head against Pytea on a fragment-fair subset, and 5/15 catches on a pre-registered unfiltered post-freeze real-PR sample while explicitly noting that this small-N result is not statistically separable from the baselines. The paper also adds a backward verifier for silent-zero-grad patterns, a hybrid TensorGuard/FakeTensor mode, and a preliminary necessary-direction correspondence to TorchDynamo guards. A recurring theme is calibrated scope: the 488-block corpus is framed as fragment-coverage rather than unconditional bug-finding, and proof-grade soundness is restricted to audited operators and verdict classes.

## Prior weakness disposition
- [RESOLVED] \textbf{Mutation testing on load-bearing handlers shows \(0/10\) kill on \texttt{conv2d} and \(0/10\) on \texttt{einsum}}... -- The repo now ships `reproducibility/mutation_kill_rate_loadbearing_v2.md`, which corrects the handler ranges and reports 20/38 for conv2d and 7/7 for einsum on the reviewer-requested comparison/arithmetic subset.
- [RESOLVED] \textbf{Of the \(185\) in-soundness verdicts on the \(488\)-block corpus, only \(36\) touch only the Lean-or-pen-paper audited footprint}... -- The current paper explicitly states `36/185` tightly inside the audited footprint and `105/185` touching at least one tested-only handler in §4.4 / Table 7 discussion.
- [RESOLVED] \textbf{The \(N{=}15\) unfiltered post-freeze headline does not separate from baselines under any standard test}... -- The paper now reports the Fisher p-values directly and frames the 5/15 result as directional rather than statistically separable.
- [PARTIAL] \textbf{The hybrid-mode "complementarity" claim ... is on a \(25\)-block stress set the authors hand-designed}... -- The paper now labels Table 4 as a stress-set result and separately reports zero gain on the 488-block corpus, but the complementarity evidence itself is still confined to the hand-designed set.
- [UNRESOLVED] \textbf{The grad-flag silent-error rate on the \emph{worst-case} construct family is \(2/8 = 25\%\)}... -- The shipped rewritten artifact `reproducibility/grad_lattice_runtime_holdout.md/json` reports `2/8` false-verified positives, yet the paper still states `0/8`, so the core concern is not resolved in the current manuscript.
- [PARTIAL] \textbf{Theorem 5 (Dynamo-guard correspondence)'s end-to-end empirical anchor is \(13\) SHAPE recompile events on \(10\) CNN-type modules}... -- The paper adds broader denominator audits and clearer scoping, but the positive end-to-end SHAPE-evidence still mainly comes from the small CNN-type set while transformer blocks remain surrogate-audited.

## Strengths
- The paper is much better calibrated than many systems papers: it clearly distinguishes `VERIFIED`/`RP` from `LW`/`ABSTAIN`, and it no longer launders the 488-block corpus into a user-visible unconditional bug-finding result.
- The formal/core-engineering split is explicit and useful: the Lean audit, theorem statements, and handler-scope partition make it possible to see where the proof actually applies.
- The 32/34 vs. 22/34 fragment-fair comparison against Pytea is a strong empirical point and materially more convincing than the synthetic or distilled checks alone.
- The post-freeze evaluation is presented more honestly than before, especially around significance and off-axis fires.
- The reproducibility layer is unusually rich; many reviewer objections are answered with concrete scripts and cached artifacts rather than prose.

## Weaknesses
- **Section 6 currently disagrees with the shipped artifact on the backward-pass limitation.** The paper states a runtime false-verified rate of `0/8` on the trainer-realistic grad-lattice harness (lines 1802–1811), but `reproducibility/grad_lattice_runtime_holdout.md/json` now reports `2/8` positive false-verifies on the rewritten self-contained runtime sample. This is not a cosmetic mismatch: it changes the practical reading of the backward verifier’s failure mode.
- **The Theorem 5 reproducibility story is internally inconsistent.** The paper’s larger-population audit says `107` candidates, `55` successful modules, and `72` in-contract INT recompiles (§4.3, lines 1560–1569), while `reproducibility/dynamo_theorem5_n200.md/json` reports `146` candidates, `67` successful modules, and `0` in-contract recompiles total. I do not know which empirical anchor the reader is supposed to trust.
- **The hybrid-mode complementarity claim is still stress-set-only.** Table 4 is explicitly a 25-block hand-designed importable falsification corpus, while §4.2 also reports that hybrid gives exactly the same `{57, 206, 225}` triple as TG alone on the 488-block corpus. So the paper has shown existence of complementary cases, not that hybrid mode helps on a natural distribution.
- **The real-public successes still do not sit cleanly inside the theorem-backed footprint.** `reproducibility/postfreeze_5catches_handler_scope.md` classifies all 5 post-freeze catches as “mixed,” i.e. each touches at least one uncovered or tested-only component; none is wholly inside the Lean+pen-and-paper audited subset. That limits how strongly the most persuasive real-bug examples support the formal claims.
- **The post-freeze baseline comparison remains underpowered in a way that matters for the headline.** The manuscript is now honest that 5/15 is not significant, but `reproducibility/postfreeze_power_analysis.md` indicates that roughly `N=80` per arm vs. FakeTensorMode and `N=187` per arm vs. Pytea would be needed for two-sided 80% power at the observed effect sizes. This leaves the “strictly above both baselines” story as suggestive rather than established.

## Questions
- Which grad-lattice result is canonical for the final paper: the manuscript’s `0/8` runtime false-verified rate, or the rewritten shipped artifact’s `2/8`?
- Which larger-population Theorem 5 audit is canonical: the paper’s `107/55/72-INT` description, or the current `dynamo_theorem5_n200.md/json` result of `146/67/0`?
- Given that all 5 post-freeze catches are “mixed” with uncovered or tested-only pieces, can the authors point to any real-public catch that lies entirely within the Lean+pen-and-paper audited footprint?
- Should the hybrid-mode claim be restated more narrowly as a stress-test existence result, rather than a general complementarity claim, since it shows zero gain on the 488-block corpus?

## Scores
Soundness: 2
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
A synchronized reproducibility pass that updates the paper to match the shipped artifacts—especially reconciling the `0/8` vs. `2/8` grad-lattice result and the stale Theorem 5 audit numbers—would likely raise my score by one point. Right now the paper is interesting and often careful, but these mismatches materially reduce my trust in the empirical calibration.

Round: 8


Changes   +0 -0
Requests  1 Premium (3m 32s)
Tokens    ↑ 1.4m • ↓ 12.1k • 1.3m (cached) • 4.4k (reasoning)
