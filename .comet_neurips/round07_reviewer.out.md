● Read round07_reviewer.md
  │ .comet_neurips/round07_reviewer.md
  └ 324 lines read

● Read README.md
  └ 352 lines read

● Get abstract (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -E "abstract|begin\{abstract\}" neurips.tex -A 60 2>/dev/null | head
  │ -80
  └ 64 lines...

● List comet and experiments_v5 dirs (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls .comet_neurips/ 2>/dev/null | head -40 && echo "---" && ls
  │ experiments_v5/ | head -30
  └ 72 lines...

● List reproducibility (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls reproducibility/ 2>/dev/null | head -40 && echo "---" && ls
  │ experiments_v5/v8/ 2>/dev/null | head -30
  └ 72 lines...

● Check stratified resample artifact (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls experiments_v5/stratified_resample_371_wilson.json 2>&1; cat
  │ experiments_v5/stratified_resample_371_wilson.json 2>/dev/null | head -50
  └ 52 lines...

● List repro and key results (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls reproducibility/ | head -50 && echo "---" && cat
  │ experiments_v5/v5_benchmark_results.json | python3 -c "import
  │ json,sys;d=json.load(sys.stdin); print(list(d.keys()));
  │ print(d.get('block_corpus',{}).get('summary',{})); …
  └ 56 lines...

● Test basic install (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && timeout 30 python3 -c "import src; print('ok')" 2>&1 | tail -5 && echo
  │ "---tests---" && timeout 60 python3 -m pytest tests/ --collect-only -q 2>&1
  │ | tail -5
  └ 8 lines...

● Check Pytea mcnemar and per-block RP (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat experiments_v5/v8/pytea_modern_mcnemar.json 2>/dev/null | python3
  │ -m json.tool 2>/dev/null | head -40 && echo "---" && cat
  │ experiments_v5/v8/per_block_user_visible_rp.json 2>/dev/null | python3 -c
  │ "import json,sys;d=json.load(sys.stdin); print(type(d), …
  └ 20 lines...

● Run headline 60bug reproducer (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && timeout 180 python3 reproducibility/reproduce_headline_60bug.py 2>&1 |
  │ tail -30
  └ 15 lines...

● Cross-check Pytea numbers in code/paper (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -rln "25/34\|p.*0.0156\|0\\.0156\|p=0\\.015" experiments_v5/
  │ reproducibility/ 2>/dev/null | head -10 && echo "---grep abstract---" &&
  │ grep -rn "32/34\|25/34" docs/paper/sections_v5/ 2>/dev/null | head -10
  └ 1 line...

● Read shell output Waiting up to 30 seconds for command output
  └ 1 line...

● Stop shell
  └ 1 line...

● Search (grep)
  │ "25/34|32/34|0\.0156|0\.001953" (docs/paper/sections_v5, neurips.tex)
  └ 17 lines found

○ Search (grep)
  │ "25/34|p=0\.0156|p ?= ?0\.0156"
  └ Timed out

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L520:680 (161 lines read)

● Check pytea fragment fair json (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat reproducibility/pytea_fragment_fair.json | python3 -m json.tool 2>&1
  │ | head -40
  └ 41 lines...

● Check Lean and 26/356 (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -E "sorry|error|Build completed"
  │ experiments_v5/v8/lean_build_v9.log 2>/dev/null | head -10 && echo "---"
  │ && cat experiments_v5/v8/per_block_user_visible_rp.json 2>/dev/null |
  │ python3 -c "import json,sys;d=json.load(sys.stdin); …
  └ 16 lines...

● Check HF 9/9 corpus (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls reproducibility/hf_extra* 2>/dev/null && cat
  │ reproducibility/hf_extra_model_family.py 2>/dev/null | head -10 && echo
  │ "---9/9 search---" && grep -rln "9/9\|naturally-occurring" reproducibility/
  │ experiments_v5/v8/ 2>/dev/null | head -10
  └ 26 lines...

● Find 26/356 figure (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -rln "26/356\|26 / 356\|empty.assume" experiments_v5/
  │ reproducibility/ 2>/dev/null | head -10 && echo "---" && cat
  │ reproducibility/cross_family_natural_bugs.json 2>/dev/null | python3 -c
  │ "import json,sys;d=json.load(sys.stdin); print(list(d.keys())[:10] if …
  └ 2 lines...

● Stop shell
  └ 1 line...

● Search (grep)
  │ "26/356|26 / 356" (experiments_v5, reproducibility, docs/paper/sections_v5)
  └ 2 lines found

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L35:130 (96 lines read)

● Check audited footprint and 356 (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat experiments_v5/audited_footprint_unconditional_rp.json 2>/dev/null |
  │ python3 -m json.tool 2>/dev/null | head -30 && echo "---356---" && grep -rn
  │ "356" experiments_v5/ 2>/dev/null | head -10
  └ 12 lines...

## Summary
TensorGuard is presented as a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that statically verifies shape and gradient-flow properties from class source. The paper claims (i) Refuted-Proof on 53/60 historical bugs, (ii) a fragment-fair head-to-head against Pytea on N=34 modern-catalogue bugs (32/34 vs 25/34, McNemar exact p=0.0156), (iii) Refuted-Proof on 9/9 naturally-occurring HuggingFace bugs across five model families, and (iv) on the 488-block real-source corpus, 0/488 unconditional Refuted-Proof, with 26/356 RP on the empty-assume_M subset (5 of which lie inside the audited handler footprint). The composition theorem is mechanised in Lean 4 over a 17-operator DSL with 17/17 closed soundness lemmas and 36 applyOp_sound_* lemmas; the soundness theorem is restricted to a 49-handler audited sub-catalogue (36 mechanised + 13 pen-and-paper). A stratified resample (n=83, seed 20260430, 8 handler families) of the 371-row Verified tied-weight subpopulation yields 2/47 silent miscarriages (Wilson 95% upper ≤ 8.37%).

## Prior weakness disposition
- [PARTIAL] Critical artifact-versus-paper discrepancy (§6 stub-mocked sample): repository contains experiments_v5/stratified_resample_371_wilson.json … -- The JSON is now committed with explicit `seed=20260430`, `n=83`, per-stratum Wilson intervals, and `k_silently_incorrect=2`; numbers in the file (Wilson hi=0.0837, 47 ok runs) match the abstract verbatim. However, the file ships only the *summary* of judgments, not the per-row labelling protocol or the human/oracle adjudication trace, so an outside reproducer cannot independently re-derive `k=2`; only the inferential step from k to Wilson interval is checkable.
- [PARTIAL] The 2/8 = 25% worst-case false-Verified rate on tied/renamed-attribute parameter sharing remains unaddressed at the mechanism level… -- The new resample reduces the worst stratum (linear-only) to 2/29 with Wilson upper 21.96%, but that bound is still >20% on the most populous stratum (134/371) and no actual mechanism change addresses tied/renamed parameter sharing. The ≤3.0% deployment-side bound continues to rest on a regex-screened prevalence estimate plus an independence assumption that the paper does not justify.
- [RESOLVED] The audited footprint improvement from 62 to 128 relies partly on 15 pen-and-paper verdicts -- Rebuttal accepted: the handler soundness table itemises each of the 13 pen-and-paper handlers to a specific T-Broadcast/T-Reduce/T-Identity instance, the einsum case has its own prop:einsum-soundness statement, and the rule-side conditions are independently checked at verification time by the Z3 obligation discharge, so the pen-and-paper step is a classification (handler→rule) on top of mechanised rule soundness rather than free-standing manual inspection.
- [RESOLVED] C2 (assume/guarantee at nn.Module boundary) still does not cite a specific proof obligation … -- Rebuttal accepted: the Lean development closes 17/17 per-operator soundness lemmas plus 36 applyOp_sound_* theorems and composes them into Subject Reduction at the module boundary; the rank-broadcast and stride-reshape side conditions are PyTorch-specific and do not appear in Findler/Meyer-style contract subtyping. lean_build_v9.log shows zero `declaration uses 'sorry'` warnings.
- [PARTIAL] The real-source headline remains 0/488 unconditional Refuted-Proof in the canonical regime -- Rebuttal partly accepted: the abstract does state the 0/488, 26/356, and 5-catch numbers in one sentence, so there is no factual misstatement. The remaining gap is substantive, not framing: on naturally-occurring real-source class code, the verifier produces zero unconditional refutations; the positive story still rests entirely on curated bug repros and on a synthesised-caller-rely regime whose CV verdicts are not unconditional. That is a real limitation of the contribution, not just a presentation issue.

## Strengths
- Reproducibility hygiene is unusually disciplined for a NeurIPS submission: `reproducibility/reproduce_headline_60bug.py` runs end-to-end in ~1 s and prints both the headline (53/60 RP) and the ablation (56/60 raw) in one invocation; the per-bug verdict pairing is committed under `reproducibility/`. The block-corpus, bug-corpus, Pytea baseline, and stratified-resample artefacts are all checked in as JSON.
- The Lean artefact is genuinely substantial: 17/17 per-operator soundness lemmas + 36 applyOp_sound_* + a composed Subject Reduction theorem on a 17-operator DSL, with sorry-free build logs (`experiments_v5/v8/lean_build_v9.log`). This is a real mechanisation, not a stub.
- The fragment-fair Pytea comparison ships both labelling conventions (b=7, p=0.0156 conservative; b=10, p=0.00195 silent-skip-reclassified) with the per-bug contingency in the appendix, so the convention choice is auditable. The matched-pair structure (Pytea-refutes ⊂ TG-refutes) is invariant under either convention.
- Honest negative reporting: the paper states the 0/488 unconditional-RP gap on real-source class code in the abstract and §6 rather than burying it. The 488-block triple is reconciled across two regimes (HCO=True vs HCO=False) with a 5-block drift attributable to bookkeeping rather than verdict flips.

## Weaknesses
- The headline real-source result is still negative. On the 488-block naturally-occurring corpus, unconditional Refuted-Proof = 0/488; the 26/356 empty-assume_M sub-count is a post-hoc restriction to blocks whose synthesised caller-rely is empty, and only 5 of those fire inside the audited handler footprint. A static verifier whose end-to-end soundness theorem covers exactly 5 catches on 488 real modules has not yet shown that its pipeline closes the gap from "works on curated repros" to "works on library source." The 53/60 historical-corpus number, by contrast, is on a curated benchmark whose construction protocol is authored by the same group.
- The stratified resample silent-miss bound is fragile in the most populous stratum. `experiments_v5/stratified_resample_371_wilson.json` reports the linear-only stratum (134 of 371 rows) as 2 silent misses out of 29 sampled, Wilson upper = 21.96%. The paper's abstract-level "≤ 8.37%" figure is the *aggregate* Wilson upper bound; on the dominant stratum the upper bound is more than twice that. The aggregate hides the per-stratum risk concentration.
- The 9/9 naturally-occurring HuggingFace bug claim is reproduced by a small set of `reproducibility/cross_family_natural_bugs*.py` scripts, but each subject is a hand-distilled extract from an upstream PR rather than a mechanically-extracted class. The selection protocol (`experiments_v5/v8/REAL_BUG_SELECTION_PROTOCOL.md` is referenced but I did not see a binding rule that pins which PRs were included vs excluded). For a 9/9 result this matters: the denominator is small enough that one or two rejected PRs would change the rate materially.
- The "5 catches inside the audited handler footprint" number does the work the paper most needs, but the audit table that pins each of the 5 to a specific Lean rule is not located in the obvious place. `experiments_v5/audited_footprint_unconditional_rp.json` is referenced but appears empty/missing in my inspection (the grep returned nothing for its expected fields), which means a reproducibility-paranoid reviewer cannot independently verify which 5 of the 488 blocks yield unconditional RP inside the mechanised footprint.
- The "feature ablation is a flat line" claim in the README — that toggling `--no-phase-check`, `--no-device-check`, `--no-grad-check`, or `--cegar-iterations` makes no difference on the aggregate corpora — is genuinely useful information, but it also undercuts the abstract's "5-theory product domain" framing. If 4 of the 5 theories never produce a verdict change on real corpora, then the contribution is really a shape verifier, with the device/phase/stride/permutation/CEGAR machinery present but inert. The paper does not visibly resolve this tension between the framing and the ablation.
- The abstract claims the soundness theorem is "restricted to a 49-handler sub-catalogue ($36$ Lean-audited $+$ $13$ pen-and-paper)" but the rebuttal text says "$15$ pen-and-paper". The paper and the rebuttal disagree on the count of pen-and-paper handlers. This is a small but reproducibility-relevant inconsistency that should be reconciled in the camera-ready.

## Questions
- For the 5 audited-footprint unconditional-RP catches on the 488-block corpus, can you provide the per-block table that pins each catch to (a) the specific Lean rule discharged, and (b) the absence of any non-audited handler in the verdict's proof?
- The pen-and-paper handler count is 13 in the abstract and 15 in the rebuttal section — which is correct? If 13, can you list the two handlers reclassified out of pen-and-paper into Lean since the prior round?
- In the linear-only stratum (n=29, k=2, Wilson upper 21.96%), what are the two silent-miss bug patterns? Are both attributable to tied/renamed-attribute parameter sharing, and if so, why does the deployment-side regex-screened ≤3.0% prevalence bound hold for them given that linear-only is 36% of the Verified subpopulation?
- The per-feature ablation `experiments_v5/feature_ablation.json` is reported as flat across the four toggles. On any single committed real-source module from `examples/check_flag_demo/`, do the device/phase/stride/permutation theories ever rule out a verdict that the shape theory alone would not? If not, what would falsify the "5-theory product domain" framing?
- The 32/34 vs 25/34 head-to-head holds Pytea fixed at its 2022 commit. Has any contemporaneous symbolic-shape baseline (e.g. `torch.compile` FakeTensor with `fullgraph=True`, which §6 reports as 34/34 on the same 34 bugs) been excluded from the table because of an applicability gate that TG also fails when subjected to the same gate? In other words, on the 481/488 blocks where torch.compile is N/A, how many would TG also be N/A on if its synthesised caller-rely envelope were disabled?

## Scores
Soundness: 3
Presentation: 3
Contribution: 2
Confidence: 3
Overall: 5

## Borderline reasons
The single change that would push the overall score from 5 to 6 is: ship a per-block audit table for the 5 audited-footprint unconditional-RP catches that pins each to a specific Lean rule and demonstrates the verdict's proof uses no non-audited handler — turning the 5/488 number from a summary statistic into a verifiable, mechanically-checked subset. That would convert the real-source story from "0/488 unconditional, with a 5-catch caveat" into "5/488 fully-mechanised end-to-end catches on naturally-occurring code," which is a qualitatively different (and publishable) claim.


Changes   +0 -0
Requests  7.5 Premium (5m 42s)
Tokens    ↑ 1.1m • ↓ 11.3k • 981.6k (cached)
