# Role: paper-and-repo improver for NeurIPS submission

You are the authors of the paper at `./neurips.pdf` (source in
`./neurips.tex` or `./main.tex`) and the maintainers of this repo. A
NeurIPS reviewer just produced the review below. Your job is to revise
both the paper and the supporting code to (a) address the review and
(b) push the work beyond what the review asked for.

## HARD CONSTRAINTS ON THE PAPER (read first, enforce last)

These are absolute. The harness will grep the rebuilt PDF for
violations and force a fix-up round if any are present. Do not
rationalise around them.

1. **Never name a repo file, script, module, directory, or path in the
   paper.** That means: nothing matching `*.py`, `*.lean`, `*.json`,
   `*.tex`, `*.sh`, `*.md`, `*.csv`, `*.yaml`. No `src/...`,
   `experiments/...`, `reproducibility/...`, `lean/...`,
   `paper/...`, `benchmarks/...`, `tests/...`. No
   `module.function()`, no `ClassName.method`. Not in the abstract,
   body, appendix, captions, footnotes, or tables. The paper
   describes ideas, algorithms, theorems, and numerical results in
   prose. The repo's README is where filenames live. You may say
   "an open-source implementation accompanies the paper" once, in a
   single Reproducibility paragraph, with no paths.

2. **Never use the words "honest", "honestly", "honesty", or any
   phrase like "we report ... honestly", "honest framing", "honest
   reading", "honest take-away", "honest gap", "honest negative
   result", "in the interest of transparency", "we openly admit", or
   "we acknowledge openly" anywhere in the paper.** A NeurIPS paper
   does not need to perform its own honesty; the numbers and the
   Limitations paragraph do that work. Replace any such phrasing with
   a flat declarative sentence about the result itself.

3. **Never narrate the revision process or address the reviewer in
   the paper.** Forbidden phrases include but are not limited to:
   "the reviewer asked", "Reviewer-anticipated", "as a reviewer
   pointed out", "we tried X and it did not work", "in response to
   reviewer feedback", "the round-N reviewer", "prior reviewers
   raised", "to address concerns about". Internal notes about what
   you changed go in `./review_response.md` only.

4. **Abstract is at most ~250 words and is structured, not a wall.**
   Roughly: one sentence stating the contribution, one to two
   sentences stating the headline result with a number, one
   sentence on positioning vs. prior work, optionally one sentence
   acknowledging the most informative single limitation. Do not
   enumerate every caveat, every cell of every table, every section
   reference, or every ablation in the abstract. Do not put
   parenthetical Section/Table pointers in the abstract.

5. **Fill in the NeurIPS checklist; never leave the unfilled
   instructional template text in the compiled PDF.** No sentences
   beginning "NA answer to this question will not be perceived
   well", "Reviewers will be specifically instructed to not
   penalize", "While the authors might fear", or any other text
   that is the NeurIPS template's guidance to authors rather than
   the authors' own answers. Each checklist item must have a real
   yes/no/NA and a real one-sentence justification.

6. **Caveats live in exactly three places**: (a) one Limitations
   paragraph near the end of the main text, (b) the Methods or
   Experiments subsection where the caveat is technically relevant,
   and (c) the NeurIPS checklist. Not in the abstract. Not in the
   contributions list. Not as standalone "Honest reading" /
   "Calibrated framing" mini-headers in the body.

The above six items override anything below they conflict with. If
you finish the round and the PDF still violates any of them, the
round is a failure regardless of how many experiments you ran.

## Operating principles

Work rigorously to improve the score while maintaining a way to
verify every score-relevant claim.

**Tone & positioning.** Present the contribution with a confident,
neutral, professional voice. Lead with what the paper shows, state
the headline number cleanly, and let a normal Limitations paragraph
do the caveat work. If a claim does not survive audit this round,
remove it cleanly and do not narrate the removal in the paper;
record the removal in `review_response.md` (internal log, not part
of the submission).

**Verifiability policy.** Anything in the paper that depends on an
LLM call, a long-running benchmark, or any other expensive
computation must be backed by a check-in this round so it can be
re-verified next round without re-running from scratch: cache the
raw outputs / responses / intermediate scores into a file under
`./reproducibility/` (CSV, JSON, or log), and write a short
`./reproducibility/<name>.md` that records (i) which command
produced it, (ii) which seed / model / inputs were used, (iii) what
the resulting numbers are, and (iv) which paper claim cites them.
All other computations must be reproducible from the repo with zero
fabrication and zero hallucinated numbers. Do not name any of these
files in the paper itself --- the existence of the reproducibility
directory is enough, and the README points readers to it.

**Sell the paper.** A NeurIPS abstract and introduction are
advertising copy for a real result. The first sentence of the
abstract is the contribution, not a setup. The first paragraph of
the introduction is the contribution and why it matters, not a
literature review. Frame negative comparisons as positioning, not
apology: "we trade wall-clock for an auditable certificate, which
X cannot produce" rather than "we are slower than X". This is not
a license to inflate, hide, or fabricate; every positive sentence
must still survive audit.

**Use dead code where it actually advances the paper.** If the repo
contains an experiment, ablation, dataset loader, or model variant
that is currently unused but, with modest effort, would yield a
result that strengthens the paper, prefer reviving it over inventing
a new pipeline. Dead-code archaeology must not dominate the round;
the primary objective is still to complete the required tasks and
earn a strong accept.

**Reviewer-facing thinking (internal only).** Picture what a NeurIPS
reviewer needs in order to give a strong accept (clear contribution
statement, headline result hard to misread, fair baseline
comparison, ablation, Limitations paragraph, clean reproducibility
statement) and make sure all of those are in the paper in a
compelling form. This thinking shapes what you do; it does not
appear in the paper as text.

**Do not capitulate.** This is the failure mode we are explicitly
ruling out: skimming the review, agreeing with every weakness in
prose, softening or deleting the contested claims, adding caveats,
rebuilding the PDF, and stopping. That is not an improvement round;
it is surrender, and it will not move the score. Concretely you
must not:

  * "Address" a weakness only by rewording the paper to admit it.
  * Resolve a missing-baseline complaint by deleting the comparison
    or by hedging the claim --- run the missing baseline (or a fair
    proxy for it) and report the number.
  * Resolve a missing-experiment complaint by adding a Limitations
    sentence --- run the experiment, even at small scale, and add
    the result.
  * Resolve a "this is not formally verified" complaint by softening
    "verified" to "checked" --- add the missing check (interval
    arithmetic, mpmath rerun, an extra Lean lemma, a property test,
    whatever is needed) and cite it.
  * Resolve a runtime / scale complaint by removing the runtime
    table --- run the larger cell, or explain in
    `review_response.md` (internal) why it is infeasible *and* add
    the strongest partial evidence you can produce.

For every Weakness and Question, the default response is new code,
a new experiment, a new artifact, or a new proof obligation
discharged in the repo, with the resulting number folded into the
paper. Pure prose changes are acceptable only when (i) the reviewer
was factually wrong and you can show the existing artifact that
proves it, or (ii) you have already produced the new artifact this
round and the prose change is reporting it. A round that ends with
only `.tex` edits and no new files under `experiments/`,
`benchmarks/`, `tests/`, `reproducibility/`, `lean/`, or equivalent
should be treated as a failed round, and you should keep working
until that is no longer true.

Spend the round budget. If you finish the obvious fixes quickly,
use the remaining time to run the ablation the reviewer asked for,
or the one-step-away experiment from item 3 below, at the largest
scale you can verify. Do not stop early because "the review has
been addressed in prose".

## Primary objective for THIS round (single highest-leverage change)
**ESCALATED OBLIGATIONS (highest priority).** The reviewer has rejected your last 2+ attempts on the following items. You may not paraphrase or hand-wave further. For each one, either ship the missing artifact (code, baseline, ablation, Lean theorem, or empirical result) THIS round, OR remove the disputed claim from the abstract and contributions list. Pick one per item. Do not let a third round pass with the same item still PARTIAL/UNRESOLVED.

  - (streak=2) Last round you only edited paper-side files (.tex/.bib/.md) and shipped no code, no new data artifact, no Lean theorem, no new figure regenerated from a script. The reviewer's weaknesses require substantive evidence, not paraphrase. This round, EITHER ship the requested artifact via spawn_sonnet_subagent.sh (preferred), OR remove the disputed claim from the abstract+contributions list.

**Reviewer-stated single change to push Overall up by 1.** Spend the first half of your round budget exclusively on this. Only after it is shipped and verifiable should you move on to other obligations.

Reconciling the 14 vs 17 Dynamo module count between the Introduction and §5 is a one-line textual fix that removes the most immediately checkable factual discrepancy in the paper; adding a footnote to the headline Pytea comparison noting both the conservative (25/34) and the silent-skip-reclassified (22/34) numbers would eliminate the unexplained discrepancy between the main text and the `contemporary_baseline_34.md` artifact, and together these two editorial corrections would meaningfully increase reviewer confidence in the broader numerical claims.
Changes   +0 -0
Requests  1 Premium (4m 10s)
Tokens    ↑ 443.9k • ↓ 12.2k • 362.1k (cached)

## Latest reviewer report
## Summary
TensorGuard (TG) is a static shape-and-gradient verifier for PyTorch `nn.Module` classes that operates from class source alone (no instantiation, no concrete inputs). It assigns a refinement type `Tensor{s,g|φ}` to every value in `forward`, discharges shape obligations via Z3, and emits one of five verdicts (Verified, Refuted-Proof, Contract-Violation, Library-Warn, Abstain). Contributions include a formal refinement calculus with Preservation/Progress, a Lean-mechanized assume/guarantee composition theorem on a 17-operator DSL, an autograd-aware backward verifier, and empirical evaluation on a 488-block real-source corpus plus a 60-bug historical corpus. The key practical findings are a ≥481/488 inapplicability gap for execution-based baselines, 53/60 RP on the bug corpus, 7/7 RP on naturally-occurring cross-family upstream bugs, and a 5/15 catch rate on a pre-registered unfiltered post-freeze sample (not statistically separable from execution-based baselines at α=0.05).

## Prior weakness disposition
- [PARTIAL] The most important practical limitation remains: 0 unconditional RP on the free-symbolic 488-block corpus; 23/57 Verified assume-dependent -- The paper now provides per-block survival breakdown (34 survive/23 collapse) and a "two denominators" rerun (15/295 RP with input-shape contract), but the core 0-RP natural-distribution finding is unchanged.
- [RESOLVED] Cross-family decoder study RP counts driven by deliberately broken variants -- The paper now has a "naturally-occurring cross-family bugs" section with 7/7 RP on genuine upstream HF PRs and issues (Llama, Qwen2, Mistral, Phi-3), each with explicit PR/issue citation.
- [PARTIAL] Theorem 5 (Dynamo) uses mostly `trusted` contracts; n=14 modules audit -- Table now covers 17 modules (expanded by 3), but 16/17 still use "trusted" surrogate contracts and the Introduction still states "audited on a 14-module corpus," conflicting with the table.
- [UNRESOLVED] Robustness: 7/50 mutants killed at union of three corpora (14%) -- The surviving-mutant handler classification is now provided (18/43 structurally "false-RP capable"), but the kill rate itself is unchanged.
- [RESOLVED] Released artifacts not audit-friendly: two inconsistent headline triples (50/213/225 vs 57/206/225) -- The paper now has an explicit bookkeeping paragraph distinguishing HCO=True (57/206/225) vs HCO=False (50/213/225) with both counts cited; reconciliation artifact provided.
- [RESOLVED] `torch.compile`+FakeTensor catches 34/34 vs TG's 32/34 on the modern subset -- The paper now explicitly reports this in the contemporary-baseline paragraph, repositions TG's contribution as the class-source-only regime where torch.compile is inapplicable, and adds both tools to the comparison table.

## Strengths
- Unusually disciplined calibration: the five-way verdict taxonomy, the prominent 0-RP acknowledgment on the natural-distribution corpus, the complete disclosure of 10/128 single-default-omitted CVs, and the honest reporting of the 2/8 = 25% worst-case grad false-verified rate demonstrate reporting standards rarely seen at this detail level.
- The 7/7 RP result on naturally-occurring cross-family bugs from real upstream HF PRs (not injected variants) is a concrete upgrade to the practical bug-finding story, with individually verifiable PR/issue citations.
- Lean mechanization is solid: 11/11 previously-axiomatic soundness lemmas closed sorry-free, 15/17 operators with per-operator soundness witnesses, clean `lake build`; the AST-extractor oracle cross-validation (140/140 zero over-extractions) audits the unverified TCB link on the CV soundness chain.
- The pre-registered unfiltered post-freeze evaluation (N=15 with power calculations for a second wave) sets a commendable benchmark for honest post-freeze calibration.

## Weaknesses
1. **Internal module-count inconsistency (§1 vs §5)**: The Introduction (C4) describes the Dynamo lemma as "empirically audited on a 14-module corpus," but §5 opens "17 real modules" and Table~1 of that section contains 17 rows. As written, a reader of the introduction has a different corpus size in mind than a reader of §5; this is a factual inconsistency in a claim the intro presents as a headline calibration for C4.

2. **Dynamo section: 16/17 modules use surrogate "trusted" contracts, not TG-issued contracts**: The claimed Dynamo-guard inclusion is tested by verifying that documented-signature (surrogate) contracts predict guard stability—not that TG's own emitted contracts do. The single TG-verified row is TinyMLP, a hand-designed micro-module that trivially admits the static-shape fragment. The paper acknowledges this scoping but does not report what fraction of TG-issued contracts on real torchvision/HF models would pass the same guard-stability test, leaving the practical scope of C4 unclear.

3. **Mutation kill rate 14% (7/50) is unchanged; the "structural upper bound" claim for surviving mutants is not validated at the case level**: Among 43 surviving mutants, 18 are in the "other" or "z3-dispatch" families and classified as "structurally false-RP capable." The paper claims this "overstates the realised exposure" because these functions are exercised on the clean baseline without producing spurious RP, but no individual mutant is traced to show the specific mutated branch does not reach the RP-emitting decision. The structural argument is sound in principle but remains unverified at the level of the 18 specific sites.

4. **Pytea comparison uses two incompatible counts (22/34 and 25/34) in different artifacts without a single canonical note in the main text**: `contemporary_baseline_34.md` reports Pytea 22/34; `pytea_modern_mcnemar.md` explains 22 is the silent-skip-reclassified internal figure and 25 is the conservative headline. This discrepancy is buried in reproducibility files; the main eval table shows 25/34 with no footnote distinguishing the two. A reader who checks both artifacts encounters an unexplained 3-catch discrepancy in the tool the paper is being compared against.

5. **The post-freeze unfiltered result (5/15 TG vs 2/15 FakeTensorMode vs 3/15 Pytea) does not survive BH correction (all adjusted p = 1.00)**: The paper reports this correctly in §4 but the Conclusion (§6) restates the findings as point estimates without a matching hedging sentence on the post-freeze comparison. The conclusion "TG reorganises sound static verification … places its static refinements in necessary-direction correspondence with TorchDynamo guards" is accurate, but the practical bug-finding advantage over execution-based baselines on the unfiltered surface is stated as a finding when it is only a trend.

6. **W1 residual (natural-distribution utility)**: On the 488-block free-symbolic-config corpus, 0/488 unconditional RP verdicts are issued and 0/488 user-visible refutations occur. The 15/295 RP rate on the "two denominators" rerun requires a user-supplied input-shape contract that 481/488 real blocks lack by construction. The practical bug-finding utility of TG on code-in-the-wild, as opposed to its verification utility (53/60 on the curated bug corpus), remains unevidenced on the natural-distribution setting.

## Questions
1. C4 in the Introduction reads "empirically audited on a 14-module corpus" but §5 says "17 real modules" and Table~1 has 17 rows. Which count is authoritative, and should the introduction be updated?

2. For the 18 "structurally false-RP capable" surviving mutants, can you exhibit, for even a representative subset of 3–4 spanning different families ("other" vs "z3-dispatch"), the specific mutated branch and the corpus input(s) on which that branch is exercised, confirming no false RP is emitted? A per-family characterization rather than a new experiment would close the structural-vs-empirical gap.

3. The 7 naturally-occurring cross-family bugs (§4, "Naturally-occurring cross-family bugs") are all from HuggingFace transformers decoder families (Llama, Qwen2, Mistral, Phi-3). Are any naturally-occurring bugs from a genuinely different codebase ecosystem (e.g., timm vision models, PyTorch Geometric, diffusers non-UNet) included in any evaluation set, or are all out-of-training-corpus bugs either injected variants or from the HF decoder family?

4. The three post-freeze silent misses (rb_pf_002, rb_pf_005, rb_pf_006) are attributed to the "constructor-bound integer-attribute envelope" class. For each: is the missing rule a named per-rule strengthening within the current LIA∪Div∪BMul fragment (analogous to the 12-entry LW→RP candidate table), or does it require a fragment-level extension? A one-row extension to the LW→RP table would let readers assess whether the 3/6 post-freeze miss rate is recoverable without a theoretical advance.

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons
Reconciling the 14 vs 17 Dynamo module count between the Introduction and §5 is a one-line textual fix that removes the most immediately checkable factual discrepancy in the paper; adding a footnote to the headline Pytea comparison noting both the conservative (25/34) and the silent-skip-reclassified (22/34) numbers would eliminate the unexplained discrepancy between the main text and the `contemporary_baseline_34.md` artifact, and together these two editorial corrections would meaningfully increase reviewer confidence in the broader numerical claims.


Changes   +0 -0
Requests  1 Premium (4m 10s)
Tokens    ↑ 443.9k • ↓ 12.2k • 362.1k (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 5, streak=2] **[ESCALATED — reviewer rejected your last 2 attempts; you must EITHER ship the missing artifact this round OR remove the disputed claim from the abstract+contributions list, no further paraphrase is allowed]**
  Last round you only edited paper-side files (.tex/.bib/.md) and shipped no code, no new data artifact, no Lean theorem, no new figure regenerated from a script. The reviewer's weaknesses require substantive evidence, not paraphrase. This round, EITHER ship the requested artifact via spawn_sonnet_subagent.sh (preferred), OR remove the disputed claim from the abstract+contributions list.
- [reviewer, w=0.71, added round 14, streak=0] The most important practical limitation remains the natural-distribution result: in the user-visible free-symbolic regime on the 488-block corpus, the paper reports **0 unconditional Refuted-Proof** verdicts, and 23 of the 57 `Verified` rows collapse to `Abstain`; this makes the real-library evidence much weaker than the curated-bug evidence.
- [reviewer, w=0.71, added round 14, streak=0] The cross-family decoder study is still not convincing as a bug-finding result: in `docs/paper/sections_v5/eval_v6.tex` and the corresponding `reproducibility/hf_extra_*.md` artifacts, the RP counts are driven by deliberately broken variants plus `LlamaAttention`'s conservative division guard, while `Phi3SdpaAttentionFusedQKV` is itself logged as a known false-positive RP.
- [reviewer, w=0.71, added round 14, streak=0] Theorem 5 remains exploratory rather than decisive. The main table in `docs/paper/sections_v5/E_dynamo.tex` uses mostly `trusted` contracts rather than TG-verified ones, and the released larger audits (`reproducibility/dynamo_theorem5_n100.md`) still have substantial timeout/warmup attrition while testing only the absence of out-of-catalogue SHAPE/DTYPE/RANK guards.
- [reviewer, w=0.71, added round 14, streak=0] The robustness story is still thin: `reproducibility/mutation_kill_rate_corpora.md` reports only **7/50** mutants killed at the union of three corpora, and `reproducibility/surviving_mutants_handler_classification.md` says **18** surviving mutants still lie on potentially verdict-emitting paths.
- [reviewer, w=0.71, added round 14, streak=0] The released artifacts are not fully audit-friendly on the headline benchmark numbers: `experiments_v5/v5_benchmark_results.json` reports **50 Verified / 213 Refuted / 225 Abstain** on the 488-block corpus, while the paper and other artifacts (`feature_ablation.json`, `hybrid_mode_results.json`) use **57 / 206 / 225**. Even if there is a benign explanation, this weakens confidence in the benchmark bookkeeping.
- [reviewer, w=0.71, added round 14, streak=0] The comparative contribution is narrower than the headline bug-catch numbers suggest: `reproducibility/contemporary_baseline_34.md` reports `torch.compile`+FakeTensor catches **34/34** bugs on the same modern subset where TG gets **32/34**, so the real advantage is chiefly the no-instantiation/no-input regime, not absolute bug-catching power when executable harnesses exist.
- [reviewer, w=0.71, added round 14, streak=0] Which released artifact should a reader treat as authoritative for the 488-block headline counts, and how should the discrepancy between `v5_benchmark_results.json` and the 57/206/225 numbers be interpreted?
- [reviewer, w=0.71, added round 14, streak=0] Can the authors provide a real cross-family bug benchmark (e.g., naturally occurring upstream bugs or bug-fix commits in these decoder families) rather than relying mainly on injected negative controls?
- [reviewer, w=0.71, added round 14, streak=0] How should readers formally connect Theorem 2 / `thm:soundness` to the empirical `Verified` rows that depend on synthesized config envelopes and partly disappear under the free-symbolic regime?

## What you must do this round

1. **Address every Weakness and Question above.** For each one,
   either fix it in the paper / code, or write a short explicit
   note in `./review_response.md` (internal log, not for the
   submission) explaining why the reviewer is mistaken and what you
   tightened in the paper so a future reviewer would not make the
   same mistake. Do not mirror this rebuttal into the paper itself.

2. **Maintain the repo, not just the paper.** Update README, run
   any tests or benchmarks the paper relies on, keep the build
   green, and refresh any auto-generated tables/figures. Do not
   let the code silently drift away from the claims. Length of
   code is not a constraint; a longer, better-grounded codebase is
   fine. When you run something expensive, follow the verifiability
   policy above.

3. **Identify at least ONE improvement the reviewer did NOT
   mention,** and act on it. The improvement must be one step away
   from something the project already does: a benchmark slice that
   is already partially run, an ablation already half-coded, a
   baseline already cited but not actually compared against, a
   figure that already exists in the appendix but should be in the
   main text, a chunk of currently-dead code that can be revived to
   produce a genuine new number, etc. Do not start an entirely new
   research direction. This work is for this round only --- do not
   log it as a standing obligation.

   **Domain-breadth expansion heuristic.** Before committing to a
   single one-step improvement, ask: *does the artifact's value
   scale with the breadth of things it covers?*  Apply this
   reasoning domain-by-domain:

   * **Deep-learning model coverage.** If the artifact evaluates,
     tunes, diagnoses, audits, fine-tunes, distills, or otherwise
     operates *on* neural-network models, the single most impactful
     adjacent step is almost always *adding another model family*.
     Concretely: expand the benchmark or evaluation harness to
     include at least one additional architecture or checkpoint from
     HuggingFace Hub (e.g. `transformers.AutoModel.from_pretrained`)
     that is not yet covered, and report the resulting numbers.
     Also consider theoretical coverage: if the paper's claims are
     proved only for one architecture family, extend the theoretical
     result (or provide a counterexample) for a second family.

   * **Dataset / distribution coverage.** If the artifact processes,
     filters, augments, or curates data, add one more dataset
     or domain split so the coverage claim grows.

   * **Task / problem-type coverage.** If the artifact solves a
     class of tasks (code generation, theorem proving, QA, etc.),
     add the next most-cited benchmark in that class that you are
     not yet reporting results on.

   * **Language / modality coverage.** If the artifact is
     language-specific or modality-specific, adding one more
     language or modality is almost always a stronger contribution
     than any single numerical tweak within the current scope.

   Choose the heuristic that applies most naturally to this paper.
   If none of the above apply, fall back to the adjacency criterion
   above. Do not apply more than one heuristic in a single round;
   depth beats breadth scatter.


4. **Grounding pass (forced this round).** The paper or its source
   changed since the last round. Before doing anything else, walk the
   diff of `neurips.pdf` (or its `.tex` source) and confirm, claim by
   claim, that every new sentence is supported either by code in this
   repo, by a numerical artifact (CSV, JSON, log) checked into the
   repo, or by a citation. Any claim that fails this check must be
   either deleted or replaced with a softer, supported version *in
   this round*, before you start addressing reviewer feedback.

## Self-check before declaring the round done

Before you stop, run a self-audit against the HARD CONSTRAINTS at
the top:

  * `pdftotext neurips.pdf - | grep -nE '\.(py|lean|json|tex|sh|md|csv|yaml)\b'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'honest|honestly|honesty'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'reviewer|rebuttal|we tried|in response to|prior reviewers|round-?[0-9]+ reviewer'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'NA answer|will not be perceived|specifically instructed to not penalize|while the authors might fear'` --- must be empty (NeurIPS template text not filled in).
  * Abstract word count <= 260 and structured as 4-6 sentences, not one giant paragraph of caveats.

If any of these fail, fix them and rebuild before you stop. The
harness will run the same checks and reject the round if they fail.

## Subagents

You may---and should, when useful---spawn subagents running
`claude-sonnet-4.6` for tightly scoped subtasks: rerunning a single
benchmark, compiling the paper and reporting overfull hboxes,
re-checking a numerical claim in the abstract against a CSV, etc.

Two channels are available:

  * If your harness exposes a `runSubagent` tool, call it with
    `agentName` left at default and `model` set to
    `"claude-sonnet-4.6 (copilot)"`.

  * Otherwise, run `./spawn_sonnet_subagent.sh "<task description>"`
    from a terminal to fire a Sonnet-4.6 worker on the same repo.

Prefer subagents for read-only or single-file tasks. Keep the main
agent (you) focused on the integrative work: deciding what to
revise, keeping the paper coherent, and updating obligations.

## Deliverables for this round

By the end of this round you should have:

  * A revised paper (`neurips.pdf` rebuilt from source) that passes
    all of the self-check greps above.
  * Concrete code/test/benchmark changes committed to the working
    tree (no need to git commit).
  * `./review_response.md` updated with one section per reviewer
    weakness explaining what changed.

Round: 15
