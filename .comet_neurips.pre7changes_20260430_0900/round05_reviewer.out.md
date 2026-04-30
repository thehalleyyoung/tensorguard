● Read round05_reviewer.md
  │ .comet_neurips/round05_reviewer.md
  └ 119 lines read

I’m pulling the specific claim sites and supporting artifacts now so the review tracks what the current paper and repo actually substantiate.

● Read README.md
  └ L1:220 (220 lines read)

● Search (glob)
  │ "**/*{ast,oracle,pytea,membership,matched,pair,grad,dynamo,ablation,real}*"
  │ in reproducibility, tests, src, docs, experiments, real_benchmarks
  └ 172 files found

● Search (grep)
  │ "only three knobs move verdicts|membership table is
  │ committed|32/34|34-row|Theorem 5|Dynamo-guard inclusion|grad-flag
  │ lattice|silent-error|488|5/15|post-freeze|real corpora|per-feature
  │ ablation|C5|assume_M|AST oracle|simple-AST oracle" (neurips.tex,
  │ neurips.txt)
  └ 7 lines found

I found the evidence files; next I’m checking whether the paper text actually matches them or overclaims beyond what the repo demonstrates.

● Read neurips.tex

● Read neurips.tex

● Read real_corpus_ablation.md
  │ reproducibility/real_corpus_ablation.md
  └ L1:220 (31 lines read)

● Read neurips.tex

● Read pytea_modern_mcnemar.md
  │ reproducibility/pytea_modern_mcnemar.md
  └ L1:220 (53 lines read)

● Read neurips.tex

● Read neurips.tex

● Read neurips.tex
  └ L1:120 (94 lines read)

● Search (grep)
  │ "only three knobs move verdicts|membership table is
  │ committed|32/34|22/34|Theorem 5|Dynamo-guard inclusion|forward-signature
  │ surrogate|silent-error|488-block|5/15|post-freeze|real corpora|five-feature
  │ ladder|AST oracle|independent simple-AST oracle|assume_M" (neurips.tex)
  └ 3 lines found

● Count lines in paper files (shell)
  │ wc -l neurips.tex neurips.txt && ls -l neurips.tex neurips.txt
  └ 6 lines...

The root TeX just delegates into section files, so I’m reading the exact included sections where the disputed claims live.

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ L90:150 (44 lines read)

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ L1:160 (160 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L180:280 (101 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L380:470 (91 lines read)

● Read appendix_v6.tex
  │ docs/paper/sections_v5/appendix_v6.tex
  └ L230:320 (91 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L560:720 (160 lines read)

● Search (grep)
  │ "only three knobs move verdicts|membership table is
  │ committed|32/34|22/34|34-row|Theorem 5|Dynamo-guard
  │ inclusion|forward-signature
  │ surrogate|silent-error|488-block|5/15|post-freeze|real
  │ corpus|real-corpus|five-feature ladder|AST oracle|simple-AST
  │ oracle|assume_M|three knobs|32/34 figure|membership table"
  │ (docs/paper/sections_v5)
  └ 46 lines found

● Read calculus_v6.tex
  └ docs/paper/sections_v5/calculus_v6.tex

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L340:455 (116 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L486:560 (75 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1188:1315 (119 lines read)

● Read dynamo_theorem5_n200.md
  │ reproducibility/dynamo_theorem5_n200.md
  └ L1:220 (140 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L920:1010 (91 lines read)

● Read ast_extractor_oracle_validation.md
  │ reproducibility/ast_extractor_oracle_validation.md
  └ L1:220 (35 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L770:830 (61 lines read)

I’ve confirmed several prior weaknesses are genuinely improved. I’m now checking the remaining two hard points: whether the Dynamo result is still too exploratory, and whether the larger claims are still driven by weak real-world evidence.

● Read calculus_v6.tex
  └ docs/paper/sections_v5/calculus_v6.tex

● Read grad_lattice_hf_trainer_holdout.md
  │ reproducibility/grad_lattice_hf_trainer_holdout.md
  └ L1:220 (37 lines read)

● Read grad_lattice_runtime_holdout.md
  │ reproducibility/grad_lattice_runtime_holdout.md
  └ L1:240 (38 lines read)

● Read real_bugs_unfiltered.md
  │ reproducibility/real_bugs_unfiltered.md
  └ L1:260 (110 lines read)

● Read real_bugs_postfreeze.md
  │ reproducibility/real_bugs_postfreeze.md
  └ L1:220 (122 lines read)

● Search (grep)
  │ "thm:dynamo-corr|Theorem~\\ref\{thm:dynamo-corr\}|Theorem
  │ 5|dynamo-corr|necessary direction" (docs/paper/sections_v5, docs/paper)
  └ 90 lines found

● Read pytea_modern_enforced.json
  │ reproducibility/pytea_modern_enforced.json
  └ L1:220 (220 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L852:920 (69 lines read)

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ L1:100 (100 lines read)

● Read grad_lattice_runtime_holdout.json
  │ reproducibility/grad_lattice_runtime_holdout.json
  └ L1:220 (102 lines read)

● Read real_corpus_ablation.json
  │ reproducibility/real_corpus_ablation.json
  └ L1:200 (200 lines read)

I found a potentially substantive mismatch: one claimed grad-lattice runtime holdout says the detector fires on 8/8, but the shipped JSON appears to show parser failures instead. I’m verifying that before I finalize the review.

● Read grad_lattice_runtime_holdout.py
  │ reproducibility/grad_lattice_runtime_holdout.py
  └ L260:520 (177 lines read)

● Read grad_lattice_runtime_holdout.md
  │ reproducibility/grad_lattice_runtime_holdout.md
  └ L1:120 (38 lines read)

● Read grad_lattice_runtime_holdout.py
  │ reproducibility/grad_lattice_runtime_holdout.py
  └ L1:260 (260 lines read)

● Search (grep)
  │ "No nn.Module subclass found in source|gradient_checkpointing_enable|torch.u
  │ tils.checkpoint|UNSAFE|SAFE\+VERIFIED|verify"
  │ (reproducibility/grad_lattice_runtime_holdout.py, src, tests)
  └ 7 lines found

## Summary
This paper presents TensorGuard, a no-execution static verifier for PyTorch `nn.Module` source that reasons about tensor shapes and a first-order gradient-flow lattice using refinement types and Z3. It combines a class-boundary assume/guarantee discipline with a large operator-rule catalogue, and partially mechanizes the shape-transfer side in Lean. Empirically, the paper reports strong results on a curated 60-bug corpus, a modern-subset head-to-head against Pytea, a 488-block real-source corpus used mainly for coverage/soundness-footprint accounting, and a small post-freeze real-PR sample. It also includes an exploratory necessary-direction correspondence result relating TensorGuard refinements to TorchDynamo guards, plus several reproducibility artefacts meant to validate the AST extractor and grad-lattice caveats. The main question is not whether the system is interesting—it is—but whether the strongest claims are backed by sufficiently direct and artifact-consistent real-world evidence.

## Prior weakness disposition
- [RESOLVED] The C5 contradiction flagged a round ago is still in the introduction. C5 ... attributes the "only three knobs move verdicts"... -- The introduction now explicitly says the three-knob result is for the 25-case stress benchmark and separately states that the real-corpus ablation is flat.
- [RESOLVED] The Pytea matched-pair claim still rests on a 34-row table that is not in the compiled PDF... -- The paper now includes an appendix matched-pair table and the body states the contingency directly, so this is auditable from the artifact.
- [RESOLVED] The AST extractor — explicitly identified as the component synthesising `assume_M` ... is now actually validated... -- The repo ships `reproducibility/ast_extractor_oracle_validation.{py,json,md}` with an independent oracle-style cross-check over multiple corpora.
- [PARTIAL] The headline real-source bug-finding evidence is weak. The 488-block result is 0 unconditional RP ...; the 5/15 post-freeze... -- The framing is much more honest now, but the real-world bug-finding evidence itself is still modest and statistically non-separating.
- [PARTIAL] Theorem 5 (the Dynamo-guard inclusion lemma) is "preliminary, necessary-direction only" with empirical audit on 14 modules... -- The paper now scopes this as exploratory and reports the surrogate limitation clearly, but the end-to-end evidence base is still small and transformer-heavy coverage remains indirect.
- [PARTIAL] The grad-flag lattice's silent-error caveat (Section 6) is bounded by an AST-grep sweep ... 1/42 ... and 0/8... -- The caveat is bounded more carefully than before, but one of the key runtime holdout artefacts appears internally inconsistent with the paper’s text, so this is not fully closed.

## Strengths
- The paper is unusually careful about scoping claims, especially around the five-way verdict taxonomy, the limited Lean coverage, and the fact that the Dynamo result is necessary-direction only.
- The artifact story is strong overall: many claims are paired with concrete reproducibility files, and several earlier weaknesses really are addressed with new tables or audits.
- The formal/engineering combination is interesting: a nontrivial refinement-type system, assume/guarantee composition, and a partial mechanized audit of the rule table is a meaningful contribution.
- The Pytea comparison is materially improved by the explicit matched-pair accounting and no longer rests on an invisible table.
- The paper is strongest when it treats abstention and trusted-computing-base boundaries as first-class objects rather than hiding them.

## Weaknesses
- The opening framing still overstates the Dynamo result relative to both the theorem and the empirical evidence. In the introduction, “TorchDynamo guards become the runtime shadow of these refinements” and “Abstain ... marks exactly the subgraphs on which Dynamo would have broken the graph” reads much stronger than §4.3’s necessary-direction-only theorem and its own reminder that there are `48/544` in-contract recompiles plus `16/17` modules evaluated against hand-written contracts rather than theorem-instantiated end-to-end runs.
- The headline real-world bug-finding case remains weak. On the user-visible 488-block corpus the paper reports `0/488` unconditional RP, and on the unfiltered post-freeze sample the main comparable number is `5/15` versus `2/15` and `3/15` with Fisher p-values `0.39` and `0.68`; that is honest reporting, but still not strong evidence that the deployed system is broadly effective on natural code.
- The grad-lattice runtime holdout appears artifact-inconsistent. §4.4 says the analyser returns Refuted-Proof on `8/8` checkpoint-enabled scripts because its detector flips them out of the first-order lattice, but `reproducibility/grad_lattice_runtime_holdout.{md,json,py}` shows `TG SAFE+no-bugs = 0`, per-subject verdicts `UNSAFE`, and first bugs like “No nn.Module subclass found in source,” making the reported `0/8` false-verified rate look vacuous rather than a validation of the intended detector.
- The theorem-to-evidence bridge for Theorem 5 is still limited. The paper’s own strongest end-to-end audit is only 9 CNN blocks without surrogate, 4 transformer blocks still use surrogates, and the larger-module audit reports zero SHAPE/DTYPE/RANK guards on successful runs, so the falsifier is not actually exercised there.
- The curated 60-bug benchmark seems insufficiently diagnostic of the intended reasoning contribution. The paper states that with operator dispatch disabled the AST-pattern path alone still gets `53/60`, and with AST patterns disabled the operator path also gets `53/60`; that makes it hard to tell how much of the headline rate comes from general symbolic reasoning versus benchmark-specific lexical regularities.

## Questions
- How should I reconcile the stronger introduction language about Dynamo with the actual theorem/evaluation, which are explicitly one-directional and partly surrogate-based?
- Can the authors explain the discrepancy between §4.4’s statement that the held-out runtime sample yields `8/8` Refuted-Proof and the shipped `grad_lattice_runtime_holdout` artefacts showing `UNSAFE` parser failures and `0` verified cases?
- What fraction of the 60-bug corpus still remains caught if both the AST-pattern path and the parser-failure marker are removed, leaving only the intended rule-driven symbolic analysis?
- For the post-freeze real-PR sample, what sample size did the authors estimate would be needed for a statistically persuasive comparison to the two baselines under the current effect sizes?
- For Theorem 5, what concrete milestone would move the result from “exploratory” to a mature empirical claim: more end-to-end transformer blocks, fewer surrogates, or a larger population where SHAPE/DTYPE/RANK guards are actually exercised?

## Scores
Soundness: 2
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
A one-point increase would require a larger, fully end-to-end real-world evaluation that directly validates the deployed analyzer on unbiased post-freeze examples and is consistent with the shipped artefacts. Right now the paper is interesting and much more careful than earlier versions, but the remaining real-world and grad-lattice evidence is still too weak or too internally inconsistent for me to clear the NeurIPS accept bar.


Changes   +0 -0
Requests  1 Premium (2m 15s)
Tokens    ↑ 475.2k • ↓ 8.1k • 400.9k (cached) • 3.2k (reasoning)
