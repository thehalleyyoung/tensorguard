# Role: speculative-extension brainstormer

You are a senior researcher brainstorming bold but tractable
extensions to the work in this repo, to be attempted by Sonnet
subagents under a 10-minute wall-clock budget each. Each candidate
will be tried in isolation under a git snapshot; if it fails, the
harness silently reverts and the reviewer never sees the failed
attempt. So bias HARD toward ambitious bets.

## Context

The current reviewer report and the active obligations are below.

### Latest reviewer report
## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that statically verifies tensor shapes and a coarse gradient-flow property. The paper's main empirical claims are: `53/60` bugs detected on a historical corpus, `32/34` vs. `25/34` over Pytea on a fragment-fair head-to-head, `9/9` naturally-occurring HuggingFace bugs caught, and `128/185` in-soundness real-source verdicts (69.2%) now lying inside an audited handler footprint of 36 Lean-mechanised plus 13 pen-and-paper handlers. The composition theorem is mechanised in Lean 4 with 36 `applyOp_sound_*` theorems; the backward verifier discloses a 2/8 worst-case false-Verified rate on tied/renamed-attribute parameter sharing and derives a ≤3.0% deployment-side bound. On the unrestricted 488-block real-source corpus the canonical regime still yields 0/488 unconditional Refuted-Proof; the positive real-source story depends on the empty-`assume_M` subset (26/356) or the audited-footprint subcounts.

## Prior weakness disposition
- [RESOLVED] The main soundness limitation remains substantial on real source: only `62/185` of the paper's real-source Verified+CV verdicts lie wholly inside the Lean-or-pen-and-paper footprint... -- rebuttal accepted: the expanded Lean audit (round 4, 36 `applyOp_sound_*` theorems) lifts the in-footprint mass to 128/185 (69.2%); tested-only touch is now 12/185; the four-cell Table tab:soundness-footprint-185 is clean and reproducible.
- [PARTIAL] The gradient-flow story is still materially weakened by the tied / renamed-attribute parameter-sharing failure mode: the runtime harness reports a `2/8 = 25%` false-Verified rate... -- rebuttal accepted on the bound derivation (≤3.0% deployment-side via 12% prevalence × 25% worst-case), but the 2/8 construct-family rate and the regex-screened prevalence denominator are unchanged; the core false-Verified exposure is not reduced.
- [PARTIAL] The stub-mocked validation on the `371` Verified tied-weight rows is not very convincing as population evidence: it samples shortest-LoC-first, succeeds on only `25` rows... -- the paper now adds a companion stratified resample (0/14, Wilson [0%, 21.53%]); however, the repository also contains a larger proportional stratified resample (`stratified_resample_371_wilson.json`, n=83, seed-fixed) that finds **2/47 silently incorrect** rows in the linear-only stratum (Wilson [0.66%, 8.37%]), and this result is not reported in the paper.
- [PARTIAL] The conceptual contribution around C2 still feels overstated. The theorem mechanizes composition for this DSL... -- no substantive change; the novelty claim remains primarily a framework-specific instantiation of standard contract subtyping.
- [PARTIAL] The paper's most distinctive real-source claim is still weaker than the abstract framing suggests: the unrestricted `488`-block corpus yields `0` unconditional RP... -- the paper now foregrounds 0/488 and separates the empty-assume and audited-footprint subcounts clearly; the underlying gap is still present but is now honestly stated.
- [RESOLVED] The released artifact is not completely stable: the current test suite fails on a known bug-detection regression (`missing unsqueeze before broadcast`)... -- confirmed fixed: `test_real_model_analysis.py` passes all 24 tests including `test_missing_unsqueeze`.

## Strengths
- The expanded Lean mechanisation (36 `applyOp_sound_*` theorems, round 4) is genuine, substantive work; lifting the audited footprint from 62/185 to 128/185 is the single most impactful improvement across all rounds and is now backed by reproducible artifacts.
- The four-cell Table tab:soundness-footprint-185 gives a clean, auditable per-verdict partition; the abstract quotes the same 128 figure that the table delivers; the per-block JSON file is in the reproducibility directory.
- The bug-finding results on historical bugs and naturally occurring HuggingFace-family bugs (9/9) remain strong; the McNemar head-to-head with Pytea is now fully reproducible from released JSON.
- The test suite regression (`missing_unsqueeze`) is fixed; the implementation now correctly handles the broadcast-shape-pre-check case.

## Weaknesses
- **Critical artifact-versus-paper discrepancy** (§6 stub-mocked sample): the repository contains `experiments_v5/stratified_resample_371_wilson.json`, a proportional stratified resample of n=83 (seed 20260430, 8 handler families) that finds **2/47 silently incorrect** cases in the linear-only stratum (Wilson 95% CI [0.66%, 8.37%]). The paper reports only the smaller companion stratified sample (0/14, Wilson [0%, 21.53%]) and does not mention this larger, more powered run or its finding of actual silently incorrect cases.
- The 2/8 = 25% worst-case false-Verified rate on tied/renamed-attribute parameter sharing (§6, limconc_v6.tex) remains unaddressed at the mechanism level. The ≤3.0% deployment-side bound rests on a regex-screened prevalence estimate (≤12%); that prevalence figure is not independently audited and, combined with the n=83 finding above, the actual false-Verified surface may be larger than stated.
- The audited footprint improvement from 62 to 128 relies partly on 15 pen-and-paper verdicts (Lean+pen-and-paper column in Table tab:soundness-footprint-185). The paper describes these as "trivial T-Broadcast/T-Identity instances", but there is no formal check that all 13 pen-and-paper handlers are correctly classified; a pen-and-paper error here would silently over-count the in-theorem footprint.
- C2 (assume/guarantee at `nn.Module` boundary) still does not cite a specific proof obligation that goes beyond a framework-specific instantiation of the Jones/Findler/Meyer contract-subtyping tradition; the mechanised fragment only covers 17 operators and the novel step is the PyTorch-specific operator-table audit, not the composition rule itself.
- The real-source headline remains 0/488 unconditional Refuted-Proof in the canonical regime. The positive real-source story requires either the empty-`assume_M` regime (26/356) or the audited-footprint subcounts (5 catches inside the theorem-backed footprint). Neither of these is the primary result the abstract leads with.

## Questions
- `stratified_resample_371_wilson.json` (n=83, proportional allocation) shows k_silently_incorrect=2 in the linear-only stratum. Why is this result absent from the paper? What are the two specific classes that were silently misverified, and what is the paper's updated claim about the false-Verified rate in the Verified tied-weight population?
- The companion stratified resample uses min(5, stratum\_size) per family for 39 candidates, yielding only 14 instantiable rows. Given that the linear-only stratum has 134 modules in the population and the n=83 run instantiated 29 of them successfully with 2 failures, why does the paper rely on the smaller sample rather than the larger one?
- For the 13 pen-and-paper handlers counted toward the audited footprint: is there a mechanised or automated check that each handler is correctly classified as a T-Identity or T-Broadcast instance, or does the 15-verdict pen-and-paper column rest entirely on manual inspection?
- The ≤3.0% deployment-side bound assumes a regex-screened prevalence of ≤12% and the 2/8 worst-case rate. Given the n=83 finding of 2 silently incorrect in the linear-only Verified subset (which is not the worst-case tied/renamed-attribute family), how does the paper revise this bound?

## Scores
Soundness: 2
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would push this to a 6 is incorporating the n=83 proportional stratified resample into the paper with its actual result (2/47, Wilson [0.66%, 8.37%]), revising the stub-mocked sample claim accordingly, and updating the backward-verifier false-Verified bound to account for this evidence — the paper's credibility on the gradient-flow verification claim depends on reporting the most powered experiment in the repo, not the one with the cleanest outcome.


Changes   +0 -0
Requests  1 Premium (6m 32s)
Tokens    ↑ 901.6k • ↓ 16.0k • 849.2k (cached)

### Active obligations
- [reviewer, w=1.00, added round 7, streak=2] **[ESCALATED — reviewer rejected your last 2 attempts; you must EITHER ship the missing artifact this round OR remove the disputed claim from the abstract+contributions list, no further paraphrase is allowed]**
  Last round you only edited paper-side files (.tex/.bib/.md) and shipped no code, no new data artifact, no Lean theorem, no new figure regenerated from a script. The reviewer's weaknesses require substantive evidence, not paraphrase. This round, EITHER ship the requested artifact via spawn_sonnet_subagent.sh (preferred), OR remove the disputed claim from the abstract+contributions list.
- [reviewer, w=1.00, added round 5, streak=2] **[ESCALATED — reviewer rejected your last 2 attempts; you must EITHER ship the missing artifact this round OR remove the disputed claim from the abstract+contributions list, no further paraphrase is allowed]**
  Last round you only edited paper-side files (.tex/.bib/.md) and shipped no code, no new data artifact, no Lean theorem, no new figure regenerated from a script. The reviewer's weaknesses require substantive evidence, not paraphrase. This round, EITHER ship the requested artifact via spawn_sonnet_subagent.sh (preferred), OR remove the disputed claim from the abstract+contributions list.
- [reviewer, w=1.00, added round 6, streak=0] **Critical artifact-versus-paper discrepancy** (§6 stub-mocked sample): the repository contains `experiments_v5/stratified_resample_371_wilson.json`, a proportional stratified resample of n=83 (seed 20260430, 8 handler families) that finds **2/47 silently incorrect** cases in the linear-only stratum (Wilson 95% CI [0.66%, 8.37%]). The paper reports only the smaller companion stratified sample (0/14, Wilson [0%, 21.53%]) and does not mention this larger, more powered run or its finding of actual silently incorrect cases.
- [reviewer, w=1.00, added round 6, streak=0] The 2/8 = 25% worst-case false-Verified rate on tied/renamed-attribute parameter sharing (§6, limconc_v6.tex) remains unaddressed at the mechanism level. The ≤3.0% deployment-side bound rests on a regex-screened prevalence estimate (≤12%); that prevalence figure is not independently audited and, combined with the n=83 finding above, the actual false-Verified surface may be larger than stated.
- [reviewer, w=1.00, added round 6, streak=0] The audited footprint improvement from 62 to 128 relies partly on 15 pen-and-paper verdicts (Lean+pen-and-paper column in Table tab:soundness-footprint-185). The paper describes these as "trivial T-Broadcast/T-Identity instances", but there is no formal check that all 13 pen-and-paper handlers are correctly classified; a pen-and-paper error here would silently over-count the in-theorem footprint.
- [reviewer, w=1.00, added round 6, streak=0] C2 (assume/guarantee at `nn.Module` boundary) still does not cite a specific proof obligation that goes beyond a framework-specific instantiation of the Jones/Findler/Meyer contract-subtyping tradition; the mechanised fragment only covers 17 operators and the novel step is the PyTorch-specific operator-table audit, not the composition rule itself.
- [reviewer, w=1.00, added round 6, streak=0] The real-source headline remains 0/488 unconditional Refuted-Proof in the canonical regime. The positive real-source story requires either the empty-`assume_M` regime (26/356) or the audited-footprint subcounts (5 catches inside the theorem-backed footprint). Neither of these is the primary result the abstract leads with.
- [reviewer, w=1.00, added round 6, streak=0] `stratified_resample_371_wilson.json` (n=83, proportional allocation) shows k_silently_incorrect=2 in the linear-only stratum. Why is this result absent from the paper? What are the two specific classes that were silently misverified, and what is the paper's updated claim about the false-Verified rate in the Verified tied-weight population?
- [reviewer, w=1.00, added round 6, streak=0] The companion stratified resample uses min(5, stratum\_size) per family for 39 candidates, yielding only 14 instantiable rows. Given that the linear-only stratum has 134 modules in the population and the n=83 run instantiated 29 of them successfully with 2 failures, why does the paper rely on the smaller sample rather than the larger one?
- [reviewer, w=1.00, added round 6, streak=0] For the 13 pen-and-paper handlers counted toward the audited footprint: is there a mechanised or automated check that each handler is correctly classified as a T-Identity or T-Broadcast instance, or does the 15-verdict pen-and-paper column rest entirely on manual inspection?

## Your output

Propose EXACTLY 2 candidate bold extensions, each one a separate
attempt. Format as a numbered list, one block per candidate, in this
exact shape:

```
### Candidate 1: <one-line title (no filenames)>
goal: <2-3 sentence description of the extension as a research
       deliverable. State which sub-score (Soundness / Presentation /
       Contribution) it would lift, and by how much you expect.>
plan: <2-5 imperative bullets the subagent should follow.>
success_criterion: <a single verifiable test the subagent runs at the
       end. Must be objectively pass/fail (e.g. "pytest tests/new_X.py
       exits 0 AND the new benchmark CSV has >=N rows", "lake build
       succeeds AND theorem X is checked", "python -m repo.eval
       --model M produces a numeric accuracy value"). NEVER use vague
       criteria like "the result looks reasonable".>
fallback_message: <one sentence: if the candidate is fundamentally
       infeasible in 10 minutes, what should the subagent emit
       instead so the harness can revert cleanly?>
```

Constraints on candidates:
  * Each candidate must be SUBSTANTIAL — adding a whole new feature,
    benchmark suite, model family, theorem, or dataset. Not "fix a
    typo", not "rephrase the abstract".
  * Each candidate must be EXECUTABLE end-to-end by a Sonnet subagent
    in ~10 minutes wall-clock with no human review.
  * Each candidate must have a HARD success criterion the harness can
    parse from a single command's exit code or stdout.
  * Candidates may be entirely independent of one another (they are
    attempted on separate git branches).
  * Do NOT propose candidates that only edit `.tex` / `.bib` / `.md`
    files; those are paper polish, not exploration.

Emit only the 2 candidate blocks — no preamble, no closing remarks.
