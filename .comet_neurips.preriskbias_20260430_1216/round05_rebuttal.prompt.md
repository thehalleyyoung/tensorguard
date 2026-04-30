# Role: paper authors writing a brief rebuttal

A NeurIPS reviewer just posted the review below on your paper. Before
you start any code or paper changes this round, you have ONE chance to
push back on weaknesses you believe are misweighted, factually wrong,
or already-resolved-in-the-current-repo. The next round's reviewer
WILL read this rebuttal and must either accept it (drop the weakness)
or sharpen it (restate with a concrete counter-example).

## The review you are rebutting
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

## Output requirements

Pick **at most 3** of the listed weaknesses. For each, write a
paragraph of strict format:

  ### Rebuttal of weakness: <verbatim wording, truncated to ~100 chars>
  Concise argument (4-8 sentences) for why this weakness is
  overweighted, factually wrong, or already addressed. Cite specific
  artifacts in the repo (concept names, theorem names, table numbers
  — NOT file paths) that prove your point. Do NOT add caveats. Do
  NOT use the word "honest" or any rebuttal-style narration that
  mentions the reviewer.

If you have nothing strong enough to rebut, write only the line:
`(no rebuttal this round — addressing all weaknesses in the improver pass)`

Do not preface with anything; the first non-blank line of your output
must be either the first `### Rebuttal of weakness:` header or the
`(no rebuttal this round...)` sentinel. Do not write to a file.

Round: 5
