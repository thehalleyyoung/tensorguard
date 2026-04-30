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
**Reviewer-stated single change to push Overall up by 1.** Spend the first half of your round budget exclusively on this. Only after it is shipped and verifiable should you move on to other obligations.

Produce a reproducible single-command artifact that emits the fragment-fair Pytea comparison (N=34, TG 32/34, Pytea 25/34, McNemar p=0.0156) from the shipped pytea_baseline_results.json, or add a standalone pytea_fragment_fair.json with per-row tensorguard_verdict/pytea_verdict fields and the 34-row subset membership documented. The abstract's second-sentence headline (32/34 vs 25/34, p=0.0156) is not independently verifiable from the current repo artifacts, which is a Soundness deduction for a paper emphasizing reproducibility.● Read (Explore agent — Assess empirical claims)
└ Completed

**Sub-score-targeted primary work (target dimension: SOUNDNESS = 3/4).** Of the four scored sub-dimensions, soundness is currently the binding constraint on Overall. Concentrate this round's non-escalated effort on raising it from 3 to 4. Concrete actions you may pick from (do AT LEAST one in full this round; sketching does not count):
  - tighten / formalize a key theorem; if a Lean / Coq / Mathematica skeleton exists, close one open lemma in this round
  - replace a proof-by-figure or sketch with a numbered theorem + proof
  - state every regularity/assumption explicitly and verify the constants in code
  - run one extra experimental seed and report the variance to defuse 'might be cherry-picked' concerns

## Latest reviewer report
## Summary

TensorGuard is a no-execution refinement-type checker for PyTorch nn.Module forward methods that verifies tensor shapes and gradient flow statakingically. The paper claims 53/60 (88.3%) refuted-proof verd inicts on a the i curated 60napplicability-bug historical gap for corpus, 32 execution-based tools (/34 vs481./488 N/A for 25/34 (p=0.0156 McNemar) torch.fx on the, line 207). The paper fragment acknowled-fairges this Pytea head (line 1518-to-head,: "corpus and 0 unconditional ref-butakedations (")rising and to 15/488 with attempts unb falsification via aind extension) on a 488-block real-source corpus.  The contribution25-block centers importable stress on a refin set (Tableement-type calculus with 4 assume/, lines 1520guarantee discipline-1528). However, a, the 60-bug corpus is mined via Lean-audited operator " table20 (+28 keyword/ queries79" with inclusion criterion " handlers with 11/11self sound-contained ness lem≤40-line CPU repromas closed" (line 649 sorry-)free — structurally bi), and ased toward bugs7 whose/ minimal7 ref reputed-proof onro matches T naturally-occurring cross-family HG's fragment.F transformer bugs.

## Prior The 4 weakness disposition silent

- [ misRESOLVED] 53/60 vsses on the bug 56/60 internal corpus (bugs 001 inconsistency in/002/006 headline/007 RP count --, Paper per consistently c Table inites 53/60 throughout paper) are not dissected:; 56/60 is no analysis explicitly of which rule scoped to was input- missingshape-supplied or regime why and the reconc SMiled inT obligation disch §arged incorre4.1 ctly. **Methodlinesological concern :** The reviewer's340-351 "corpus. and fr

- [PARTIALaming are] co-designed" critique CEGAR and phase-check ship but (REVIEW are_V architect8.urally nonmd line-functional as 97 described) is partially addressed by the diversity analysis (W -- Rebuttal does not address;# no new artifact6 resolved) evidence that but not by CEGAR/phase-check forward to verify a_ causallyforward; independent feature_ bug sourceabl. Aation.json held meta-out note corpus from remains a different ecosystem unchanged (.

- [RESOLVEDe.g., TensorFlow,] Mutation-testing JA kill rate onX) or load a non-keyword-bearing handlers is- lowm without corpusined set would strengthen gener extension -- Rebuttal acceptedalizability claims:. mutation

###_ (4) Suggested Scorekill_rate_loadbearing_v2.md documents Range conv2d 53% and einsum

**Sound 100% kill rates on comparison+arithmetic subset (ness/bothTechnical >50% Quality:** 5); the /1018 (up- fromcase v extension7's corpus is the 3/10). appropriate Preservation methodology.

- [PARTIAL/Progress proofs are complete in] Theorem 5 ( appendix;Dyn Leanamo audit) fals coversification 28 operators pred sorryicate-free. Gains is vacu:ously satisfied diversity on the large-corpus analysis (K=406), aud witnessits checks -- Rebuttal argues (118 0/128), post-freeze  generSHAPEalization (10/ DTYPEreal PR/RANKs). Losses persist guards is: 0 itself unconditional RP on  the488-block corpus, measurement substit, butution lemma sket the necessaryched,-only direction ( 44 tested-only handlers,converse  Theorem 58. necessary8-only with% in 8-contract re.8% in-contract recompile falsifyingcomp practicalile utility gap.

**Presentation) limits the theorem/Clarity:** 7/10.'s practical Calib falsratedification surface language ("necessary;-direction the vac only", "testedu-only", "0ity fr unconditional RP") isaming is def exemplusedary. Table but 5 the (handler scope calibration concern remains.

- [RESOLVED] No single command reproduces the headline), verdict taxonomy 53/60 RP figure (RP/CV/LW/A), -- Rebuttal accepted: run and explicit caveats (parameter_verdict_r-eclsharingassification.py bug documented in reproduc classibility appendix; the two prevalence -call≤12 pipeline%, ( line 110run) are best_v-in5_benchmark.-class for honesty. Depy → verdict_reclassification.json) is explicitlyductions described and: abstract/intro artifact still-check overable.

- [RESOLVED] The-frame 0/ Dynamo correspondence488 and uncon "ditional RP on realLean-audited" scope relative source is much to body more hed damges.aging to

**Contribution the contribution/ than the paper allowsNovel -- Rebuttal accepted: unbty:** 5ind_handler_488_run./10. The calcjson documentsulus is standard ( LViq=55, RP=15, A=418) after unbind+uidTypes/Pytsubea lineclass-recogn; gradizer extension,iant exceeding the 12/-flag extension is non-trivial but78 narrow LW→ (RP8/ ceiling8 canonical bugs cited in prior, review 0;/50 fals FifiP,ability criterion but met small. N

-). Assume [RESOLVED] The headline/guarantee composition 53 is operator/60 RP on-agnostic ( theLean historical mechan bugized for corpus is partially attribut 17able to category ops-keyword, line AS 99T pattern- matches -- Rebuttal accepted: ast106) but the_pattern_disabled_60bug. theoremjson shows full itself_rp=53, disabled_rp=53,  is routine0 bugs. Real caught novelty: only engineering by bundle ( AST-488pattern path-block corpus,; bug diversity_corpus_loo_handler.json documents non-zero per analysis,-category verdict taxonomy, RP drop Dyn on everyamo audit). load-bearing handler removal This is a strong *, confirtoolming operator* contribution but incre-dispatchmental * responsibility.P

-L [RESOLVED] The grad latt* contribution.

**Empirice isical Rigor:** 6 acknowledged/10  sil(up from vently incorrect7's 4/10). under Gains parameter: diversity K=406 refutes- nearsharing-under-renamed-attribute -- Rebuttal accepted: backward_-duplicate concern, post-freeze 10-param_sharing_audit.json documentsbug  corpus0/6 false-verified rate, on witness BERT/ checks, modernGP PytT-2/ea subsetT (325//34BART/R vsoBERTa/ 25minimal tied-weight rep/34, line 1ros181). Losses: 60; TG-bug corpus small conserv and keyword-mined, 4 silent misses undissected, Dynamo audit on 14 modules with 4 transformer surrogates, hybrid-mode falsification on 25-block stress set is existence proof not distributional.

**Overall:** **6/10 (borderline weak accept)**. The vatively yields lattice top8 revision substant (ively addresses corpusSAFE diversity (#_NO_BUGS6  or abstain),resolved) and calc not silentulus ri SAFEgor (#7 partial). The tool_NO_BUGS-when is- productionunsafe-grade (.

## Strengths

- **Calibrated empircontentical reporting-.** The 0/488 unconditional-RP gap on real-source blocks is acknowledgedaddressed up corpora, pinned versions front, Lean ( export§4.1 line gate 68) and the calib), LratedW verdicts are distinguished reporting from RP/CV soundness claims, and the 12 is exempl/78 LW→RP candidateary. The core table limitation is falsifiable (three already measured- persists: **flTheoremipped post-unb 2'sind).

- **Rebuttal reb soundness guarantee hasuts concretely.** The unbind 488-run artifact (15/488 RP), zero AST-pattern-disabled audit empirical footprint on unconditional refutations on the (0 488-block real/60 marginal contribution), handler corpus**. Every-LOO per real-category drops, and tied-library verdict is CV-weight backward ( auditconditional on synthesized assume_M) or LW (outside theorem (0/6 false-verified) are all one-command reproduc scope). The 56/60 historical-bug performance is strongible with shipped but on JSON a artifacts ke small, structurally-yed to reviewer-specific weakbiased corpus.nesses.

- **Cross This is a defens-family naturallyible accept for-occurring bug N validationeurIPS *.** The 7/7 if* framed as a toolsRP on upstream H/F transformer PRs/benchmarks contributionissues (;Llama/Qwen2/Mistral the theoretical apparatus/Phi-3) with (Preservation/Progress, Lean public audit) is solid PR but well numbers cited-t per row is a held-outrodden. distribution validation that the  Recommend:60 accept-bug cu with minor revisions to derated corpus does-emphasize not directly Theorem provide 5.

- **Lean bid mechanirectionality inization abstract/ scope is explicitlyintro, add bounded 1.** The paper restricts-paragraph Theorem 2 autopsy sound perness to the 44 silent- miss, and forehandlerground Table 1's Cat_sound sub 0/-catalogue128 where preservation/78  argumentssplit in are closed abstract (.Lean

** or penConfidence-and-paper), and separately:** 5/5 reports (this is squ the 28-arely my areahandler Lean-audited subset; I have implemented refin with 11/11 sorry-free lemement-typedmas; shape this is a sound checness-calibkers andration reviewed win over over-claiming.

## Pytea/LiquidTypes/ Weaknesses

- **Pytea baseline comparison headline (32Hasktorch literature)./34 vs 25/34, p=0.0156) is not reproducibly extracted from the shipped pytea_baseline_results.json.** The JSON per_input array does not populate tensorguard_verdict/pytea_verdict fields as expected; manual extraction returns 0 refuted for both tools. The paper cites fragment-fair N=34 and McNemar p=0.0156 (§4.1 line 485) but the shipped artifact does not contain this breakdown. The footnote on line 487 states "The 25/34 figure is the conservative convention"The theory pass is done; I’m but the artifact does pulling that not document which  result and then I’ll finish the review using the repo evidence already gathered.34 rows constitute the fragment-fair subset or how the 7-point gap is derived.

- **LOO handler audit methodology does not match reviewer question.** The bug_corpus_loo_handler.json shows all LOO runs yield RP=53 (identical to full_pipeline RP=53);

● Read (Explore agent — Assess theory claims)
  └ Completed

 what changes is the err count (7 errors introduced), not the RP count. The rebuttal claims "non-zero per-category RP drop on every load-bearing category" but the JSON shows RP=53 across all LOO runs. The per-category RP numbers in the LOO runs match the full_pipeline per-category breakdown exactly (attention_dim 4, broadcasting 7, conv 6, etc.). The actual sensitivity signal is the shift from silent→err when handlers are removed, not RP degradation.

- **The 128 CV caller-rely satisfiability claim (92.2% joint-realisable, 10/128 single-default-omI have the theory side. I’m pulling the empirical pass one last time so the final review reflects the current rebuttal artifactsitted) confl, not just the paper text.ates two distinct notions.** §4.1 lines 114-130 states "Zero assume_M is unwitnessed" (0/128 unwitnessed) but then "10/128 single-default-om

itted"; the text clarifies these are distinct definitions but the headline 92.2% (118/128) is the single-default metric, not the caller-pattern-witness metric. The 0/128 unwitnessed claim is therefore definitionally weaker than the 92.2% headline suggests, since it only requires *some* caller pattern to satisfy assume_M, not the specific default config.

- **AST extractor cross-validation audit is self-consistency, not third-party soundness.** §4.1 lines 140-196 documents 140/140 deployed-extractor ⊆ oracle-extractor on symbolic_config_attrs and 20/20 hand-label agreement, but explicitly states "both the deployed extractor and the oracle were implemented by the authors against the same surface specification, so this is a self-consistency cross-check, not a third-party soundness audit." The residual TCB exposure on systematic extractor-vs-spec mismatch is preserved as a Limitations item, which is calibration-honest but leaves the assume_M synthesis in the TCB without independent corroboration.

- **Headline mutation kill rates (53% conv2d, 100% einsum) are on the union corpus with targeted extension.** The mutation_kill_rate_loadbearing_v2.md documents 20/38 (conv2d) and 7/7 (einsum) on comparison+arithmetic mutations over the 60-bug + 18-case load-bearing union corpus. The paper §4.1 line 62 cites "mutation kill rate on conv2d" without prominently surfacing that the 18-case extension was author-designed to exercise the conv2d/einsum arithmetic internals. The full kill rate (including boolean-op flips) is 42% conv2d, dropping below 50%; the comparison+arithmetic subset (53%/100%) is the reviewer-asked subset but the full rate is more representative of handler robustness.

## Questions

- What is the reproducible command or script that emits the fragment-fair 34-bug subset and the 32/34 vs 25/34 per-tool breakdown cited in the abstract and §4.1 line 485? The shipped pytea_baseline_results.json does not contain this breakdown.

- The bug_corpus_loo_handler.json shows RP=53 across all LOO runs. How does this support the rebuttal claim of "non-zero per-category RP drop on every load-bearing category"? Does the sensitivity signal lie in the silent→err transition rather than RP degradation?

- The AST extractor cross-validation audit (140/140 subset, 20/20 hand-label agreement) is author-implemented oracle vs author-implemented extractor. Is there a plan to solicit third-party re-implementation or independent ground-truth labeling to address the systematic-design-mismatch TCB exposure explicitly preserved in the Limitations?

- What is the mutation kill rate on the 60-bug corpus alone (without the 18-case targeted extension) for the full mutation class enumeration, not just the comparison+arithmetic subset? The paper's headline 53%/100% is on the union corpus with extension; the regressor-alone baseline provides a calibration anchor.

## Scores

Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons

Produce a reproducible single-command artifact that emits the fragment-fair Pytea comparison (N=34, TG 32/34, Pytea 25/34, McNemar p=0.0156) from the shipped pytea_baseline_results.json, or add a standalone pytea_fragment_fair.json with per-row tensorguard_verdict/pytea_verdict fields and the 34-row subset membership documented. The abstract's second-sentence headline (32/34 vs 25/34, p=0.0156) is not independently verifiable from the current repo artifacts, which is a Soundness deduction for a paper emphasizing reproducibility.● Read (Explore agent — Assess empirical claims)
  └ Completed

## Summary

This paper presents TensorGuard, a no-execution static checker for PyTorch `nn.Module` forward methods that uses refinement types over symbolic shapes plus a simple grad-flow lattice to analyze class source without instantiation or tracing. The paper claims a refinement-typed calculus with preservation/progress, an assume/guarantee discipline at module boundaries, a partially mechanized Lean audit of the operator-rule table, and a necessary-direction correspondence between TensorGuard refinements and TorchDynamo guards. Empirically, it reports 53/60 `REFUTED-PROOF` verdicts on a curated historical bug corpus, a 32/34 vs. 25/34 fragment-fair comparison against Pytea, and a calibrated real-source story on 488 library blocks in which the default free-symbolic regime yields 0 unconditional `REFUTED-PROOF` verdicts while a stronger contract-supplied rerun reaches 15/488 after the implemented `unbind` extension. The paper is unusually explicit about what is and is not inside the soundness footprint: Theorem 2 covers `RP`/`CV` only, Theorem 5 is necessary-direction only, and much of the analyzer remains in the trusted computing base. The net contribution is therefore not “full static verification for PyTorch,” but a carefully scoped refinement-typed checker with a partially mechanized operator catalogue and unusually honest calibration.

## Prior weakness disposition

- [PARTIAL] **Conceptual novelty over Pytea is thinner than the contribution list suggests.** C1's refinement-typed calculus is a presentation reorganisation... -- The genuinely distinctive pieces are still the joint shape+grad analysis, the no-instantiation class-source regime, and the calibration machinery; the underlying constraint-based shape calculus itself still feels less novel than the contribution list suggests.
- [RESOLVED] **Headline 5/15 on the post-freeze unfiltered sample is statistically indistinguishable from the execution-based baselines** ... -- The current text explicitly says 5/15 is “point above ... not a separation,” reports Fisher p-values (0.39, 0.68), and no longer tries to sell this row as a statistically decisive win.
- [PARTIAL] **The headline 53/60 RP on the historical bug corpus is on a corpus assembled by keyword search and curated by the authors...** -- Rebuttal accepted that disabling the AST-pattern path leaves the headline unchanged at 53/60, but the corpus is still keyword-mined and author-curated with knowledge of the operator catalogue, so independence from TG’s design remains only partially established.
- [PARTIAL] **The 0/488 unconditional RP on real source is much more damaging to the contribution than the paper allows.** ... -- Rebuttal accepted that the implemented `unbind` rerun lifts the strict-denominator contract-supplied result to 15/488, but the abstract’s default user-visible free-symbolic regime is still 0/488, so the practical real-source gap remains central.
- [PARTIAL] **Mutation kill rate on `conv2d` is 0.42 on the load-bearing extension corpus** ... -- Improved but not fully retired: the comparison/arithmetic-only subset rises above 50%, yet the full load-bearing figure remains 0.42, so the robustness story is still only moderate.
- [PARTIAL] **Theorem 5 is now explicitly the necessary direction only, with an 8.8% in-contract recompile rate quantifying the converse gap.** ... -- The calibration is much better, but the result is still a narrow necessary-direction correspondence audited on a small population with surrogate transformer cases, not a strong runtime-equivalence theorem.
- [PARTIAL] **Soundness scope is fragile in a way the abstract does not surface.** Theorem 2 covers RP and CV; CV soundness is conditional... -- Improved: the abstract now surfaces the audited-handler footprint, but `CV` still depends on synthesized `assume_M`, and only 118/128 rows are witnessed under the single-default check, so the end-to-end scope remains fragile.

## Strengths

- The paper is notably honest about scope: it cleanly separates `RP`, `CV`, `LW`, and `ABSTAIN`, states Theorem 5 as necessary-direction only, and surfaces the audited-footprint counts directly in the abstract.
- The rebuttal is concrete rather than rhetorical. I accept the `unbind` rebuttal (15/488 on the same denominator) and the AST-pattern rebuttal (53/60 unchanged with that path disabled) because both are backed by explicit shipped artifacts.
- The formal side is materially stronger than in many systems papers: the preservation/progress development is now explicit, the operator audit is real, and the trusted-computing-base boundary is spelled out instead of blurred.
- I also accept the tied-weight rebuttal: the shipped backward audit on six HF/minimal tied-weight subjects removes the strongest version of the “silently incorrect under parameter sharing” accusation.

## Weaknesses

- The contribution framing in Section 1 still overstates conceptual novelty relative to Pytea-style constraint-based shape analysis. The new substance is the joint shape+grad layer and the calibration/audit package; C1 by itself reads more like a careful repackaging than a major conceptual jump.
- The historical 60-bug corpus remains author-mined and author-curated. The AST-pattern-disabled result removes one specific confound, but it does not remove the broader concern that the benchmark source and inclusion rule were designed with knowledge of TG’s operator surface.
- The real-source applicability gap remains load-bearing. In the paper’s default user-visible free-symbolic regime the 488-block corpus still yields 0 unconditional `REFUTED-PROOF` verdicts, while the stronger 15/488 number appears only after supplying an input-shape contract and enabling added extensions.
- The mutation-robustness story is still middling for a load-bearing handler: `reproducibility/mutation_kill_rate_loadbearing_v2.json` reports `conv_channel_mismatch` at 0.42 on the full load-bearing corpus, with “above 50%” holding only on a restricted mutation subset.
- Theorem 5 remains a narrow correspondence result. The paper itself quantifies an 8.8% in-contract recompile rate, and four transformer cases still require documented forward-signature surrogates rather than full end-to-end instantiation.
- Theorem 2’s empirical footprint on real-source code is still limited. The abstract’s own counts show only 11/57 `VERIFIED` and 25/128 `CV` real-source verdicts lying wholly inside the Lean-or-pen-paper audited footprint, while `CV` soundness remains conditional on synthesized caller assumptions.

## Questions

- What, precisely, do the authors view as the conceptual novelty over Pytea: the calculus itself, the grad lattice, the no-instantiation regime, or the audit/calibration package? Right now those are bundled together too aggressively.
- Can the authors provide a benchmark source that is not keyword-mined by the same team that designed the operator catalogue, so that the 53/60 result is less entangled with corpus construction?
- Since the practical story changes sharply between 0/488 in the default regime and 15/488 in the stronger contract-supplied rerun, which regime do the authors actually want readers to treat as the main deployment model?
- For Theorem 5, would the authors consider making the fully end-to-end non-surrogate subset the explicit headline, with the surrogate transformer cases clearly demoted to supporting evidence?

## Scores

Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons

The single change that would raise my score by one point is a theorem-backed real-source headline in the default user-visible regime: even a small but nonzero set of unconditional `REFUTED-PROOF` catches on library code, all inside the Lean/pen-paper audited footprint. That would align the paper’s strongest formal claim with its practical surface instead of leaving real-source success concentrated in a stronger rerun/conditional regime.

Round: 5


Changes   +0 -0
Requests  1 Premium (6m 13s)
Tokens    ↑ 1.8m • ↓ 32.7k • 1.5m (cached) • 6.4k (reasoning)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 5, streak=0] The contribution framing in Section 1 still overstates conceptual novelty relative to Pytea-style constraint-based shape analysis. The new substance is the joint shape+grad layer and the calibration/audit package; C1 by itself reads more like a careful repackaging than a major conceptual jump.
- [reviewer, w=1.00, added round 5, streak=0] The historical 60-bug corpus remains author-mined and author-curated. The AST-pattern-disabled result removes one specific confound, but it does not remove the broader concern that the benchmark source and inclusion rule were designed with knowledge of TG’s operator surface.
- [reviewer, w=1.00, added round 5, streak=0] The real-source applicability gap remains load-bearing. In the paper’s default user-visible free-symbolic regime the 488-block corpus still yields 0 unconditional `REFUTED-PROOF` verdicts, while the stronger 15/488 number appears only after supplying an input-shape contract and enabling added extensions.
- [reviewer, w=1.00, added round 5, streak=0] The mutation-robustness story is still middling for a load-bearing handler: `reproducibility/mutation_kill_rate_loadbearing_v2.json` reports `conv_channel_mismatch` at 0.42 on the full load-bearing corpus, with “above 50%” holding only on a restricted mutation subset.
- [reviewer, w=1.00, added round 5, streak=0] Theorem 5 remains a narrow correspondence result. The paper itself quantifies an 8.8% in-contract recompile rate, and four transformer cases still require documented forward-signature surrogates rather than full end-to-end instantiation.
- [reviewer, w=1.00, added round 5, streak=0] Theorem 2’s empirical footprint on real-source code is still limited. The abstract’s own counts show only 11/57 `VERIFIED` and 25/128 `CV` real-source verdicts lying wholly inside the Lean-or-pen-paper audited footprint, while `CV` soundness remains conditional on synthesized caller assumptions.
- [reviewer, w=1.00, added round 5, streak=0] What is the reproducible command or script that emits the fragment-fair 34-bug subset and the 32/34 vs 25/34 per-tool breakdown cited in the abstract and §4.1 line 485? The shipped pytea_baseline_results.json does not contain this breakdown.
- [reviewer, w=1.00, added round 5, streak=0] The bug_corpus_loo_handler.json shows RP=53 across all LOO runs. How does this support the rebuttal claim of "non-zero per-category RP drop on every load-bearing category"? Does the sensitivity signal lie in the silent→err transition rather than RP degradation?
- [reviewer, w=1.00, added round 5, streak=0] The AST extractor cross-validation audit (140/140 subset, 20/20 hand-label agreement) is author-implemented oracle vs author-implemented extractor. Is there a plan to solicit third-party re-implementation or independent ground-truth labeling to address the systematic-design-mismatch TCB exposure explicitly preserved in the Limitations?
- [reviewer, w=1.00, added round 5, streak=0] What is the mutation kill rate on the 60-bug corpus alone (without the 18-case targeted extension) for the full mutation class enumeration, not just the comparison+arithmetic subset? The paper's headline 53%/100% is on the union corpus with extension; the regressor-alone baseline provides a calibration anchor.

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

Round: 5
