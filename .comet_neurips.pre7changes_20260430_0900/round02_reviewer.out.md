● Read round02_reviewer.md
  │ .comet_neurips/round02_reviewer.md
  └ 121 lines read

● Check working tree status (shell)
  │ git --no-pager status --short
  └ 262 lines...

I’ve got the review prompt; next I’m pulling evidence from both the current paper and the repo so the dispositions and score are grounded in what’s actually present now.

● Explore Inspecting paper claims
  └ Agent started in background with agent_id: paper-claims. You'll be notif...

● Explore Checking repo evidence
  └ Agent started in background with agent_id: repo-evidence. You'll be noti...

I’m gathering paper-side and repo-side evidence in parallel so the review reflects the current artifact rather than the prior round.

**(1) Factual summary (4–6 sentences).**  
TensorGuard is a no-execution, refinement-type checker for PyTorch `nn.Module` `forward` methods that statically tracks tensor shapes and gradient-flow flags from class source, without instantiation or tracing (Abstract, lines 11–25). It claims soundness for proof-grade verdicts via a refinement calculus, plus an assume/guarantee module-boundary discipline; only part of the operator rule table is Lean-audited, while the analyser implementation and related runtime machinery remain in the TCB (Abstract lines 19–24; Contributions lines 95–124). Empirically, it reports 53/60 on a historical bug corpus, 32/34 vs 22/34 against Pytea on a fragment-fair subset, and a directional but non-separable 5/15 result on a pre-registered real-PR sample (Abstract lines 14–18; Table 1 / Section 4.1 lines 693–700, 850–856). The paper also frames a preliminary Dynamo-guard inclusion result as one-directional and limited, not an equivalence theorem (Contributions lines 104–112; Theorem 5 framing). It openly reports limitations: the 488-block real-source corpus yields zero unconditional `REFUTED-PROOF`, many `CONTRACT-VIOLATION` verdicts rely on synthesized caller-rely assumptions, and the first-order grad lattice is known to miss parameter-sharing/renaming cases (lines 693–715; 104–111).

**(2) Prior weaknesses status.**

1. **Headline real-world bug-finding result is weak (Table 3 N=15 etc.).** **PARTIAL** — the paper now explicitly downgrades the N=15 post-freeze result as “directional” and “not statistically separable” (Abstract lines 17–18; lines 693–700), but the real-world catch headline is still small and not a strong bug-finding claim.  
2. **Lean mechanization claim does not extend to soundness of deployed verifier.** **RESOLVED** — it now says the Lean audit covers only the rule table, while the analyser implementation, AST extractor, backward verifier, and Z3 dispatch remain in the TCB (Abstract lines 20–24; Contributions lines 120–124).  
3. **Theorem 5 Dynamo-guard correspondence over-scoped relative to evidence.** **RESOLVED** — Theorem 5 is explicitly framed as “necessary direction only,” “preliminary,” audited on 14 modules, with 4 transformer blocks using surrogates, and “not” a two-directional correspondence (Contributions lines 104–112).  
4. **128 ContractViolation verdicts depend on synthesized caller-rely envelope with only single default config realisability checked.** **PARTIAL** — the paper now says CVs are sound only under synthesized caller-rely assumptions and reports a single default `*Config()` witness check with 118/128 witnessed (lines 701–715), but that still leaves the single-default-realisability limitation in place.  
5. **First-order grad-flag lattice silently incorrect on parameter-sharing-under-renamed-attribute, prevalence bounded ≤12% etc.** **UNRESOLVED** — the paper explicitly admits the lattice is “silently incorrect” there and says prevalence is ≤12% (Contributions line 104; also Section 6 references), with no fix.  
6. **Two of three discriminative features in Table 5 are no-ops on real corpora.** **RESOLVED** — the paper now restricts the contribution to the three discriminative knobs and explicitly says the unused CEGAR loop and always-satisfiable phase encoder are not claimed as contributions (Contributions lines 114–119).  
7. **Presentation caveats too dense.** **UNRESOLVED** — the paper still front-loads multiple caveats in the abstract and contributions and repeatedly interleaves claims with exceptions (e.g. lines 17–24, 104–124, 693–715).  
8. **32/34 vs 22/34 McNemar result lacks per-bug agreement table.** **UNRESOLVED** — the paper states the 32/34 vs 22/34 result and McNemar p=0.00195 (Abstract line 16; lines 577–580, 693–700), but I do not see a per-bug agreement table in the current text excerpt.

**(3) Additional presentation / claim-scope issues.**  
- The abstract and contributions are very caveat-dense, mixing headline claims with scope limitations in nearly every sentence (Abstract lines 14–24; Contributions lines 104–124).  
- The 488-block real-source corpus is framed as a fragment-coverage measurement rather than a bug-finding result because it yields 0 unconditional `REFUTED-PROOF` (lines 693–700), which weakens the “real-world” narrative.  
- The paper repeatedly depends on synthesized config/caller-rely envelopes for its stronger verdicts (lines 696–715), so the strongest empirical numbers are conditional rather than unconditional.● Read (Explore agent — Inspecting paper claims)
  └ Completed

Concise repo audit:

1) Prior weaknesses 1–8
- W1 corpus-wide unconditional RP: **still weakened/qualified**. `review_response.md:10-25` says 488-block corpus has **0 unconditional RP** under no-assume, with 128 CV + 78 LW. Supported as a caveat, not a headline. Related artifact: `reproducibility/no_assume_subset_488.md` / `reproducibility/per_block_user_visible_rp.md`.
- W2 Dynamo correspondence audit: **mostly supported, but narrowed**. `review_response.md:30-52` adds end-to-end `experiments_v5/v8/dynamo_e2e/run_dynamo_e2e.py` + `.../dynamo_e2e_results.json`; 5/5 subjects SAFE. But paper-level claim is explicitly downgraded to **necessary-direction only** (`neurips.txt:104-112`).
- W3 Lean sorry-freedom: **weakened by a live mismatch**. Paper/README say `11/11` and “sorry-free” (`neurips.txt:20-24`, `202-2030`; `README.md:249-251`), but `lean/TensorGuard/Extended.lean:85-108` still contains the live `sorry` at line **94** (confirmed by `reproducibility/lake_build.md:10-31`). This is the biggest soundness discrepancy.
- W4 Lean boundary precondition test: **supported**. `reproducibility/lean_precondition_boundary_test.md:14-20, 26-53` reports **6913 off-envelope samples, 0 silent-through** (6913/6913 safe failure/disagreement).
- W5 LOO holdout: **supported, and the prior no-op is fixed**. `reproducibility/per_rule_ablation_60bug.md:3-53` says old LOO was a no-op; new per-handler-family ablation shows baseline **53/60**, with real deltas (e.g. `view_reshape_total_size` and `broadcasting` each to **46**, `conv_channel_mismatch` **47**).
- W6 10/10 hand-distilled bugs: **weakened as a coverage check, not a headline**. `review_response.md:104-113` and `neurips.txt:23-24` frame it as catalogue-coverage only; the stronger real-public numbers are `7/10 @≥0.99 + 1/10 @≥0.80` (`neurips.txt:2026-2029`).
- W7 backward verifier / grad-flag limitation: **supported as a real limitation**. `neurips.txt:104-107, 2007-2019` says first-order grad lattice is silently incorrect for parameter sharing/checkpointing; prevalence estimates are backed by audits, e.g. `neurips.txt:2007-2019` and `review_response.md:117-125`.
- W8 Theorem 2 reach gap: **supported**. `reproducibility/handler_scope_per_block.md:12-23, 30-47` gives the explicit split: **11/57** verified verdicts only touch Lean-or-pen-paper handlers; **33/57** touch tested-only handlers.

2) Key discrepancies vs paper claims
- **Lean coverage**: paper/README say `11/11 sorry-free` and “entire tree sorry-free,” but the current source still has a live `sorry` in `lean/TensorGuard/Extended.lean:94`; `reproducibility/lake_build.md:13-30` explicitly names it as the only executable sorry.
- **Dynamo correspondence**: paper/README now correctly limit this to a **necessary-direction** lemma; the stronger “signature-trusted 17-module” story is replaced by the new end-to-end 5-module audit (`review_response.md:30-52`).
- **ContractViolation / caller-rely realizability**: supported by `reproducibility/cv_caller_rely.md:21-40` — **128/128 CV verdicts classified, 0 unwitnessed**, bucketed as 90 symbolic-config-only, 26 empty, 12 no-own-init. This is good evidence, but it is still pattern-matching on `__init__`, not a runtime witness construction.
- **Grad-flag limitation**: the paper’s caveat is backed by concrete held-out evidence in `neurips.txt:2007-2019` (1/42 silent-error-positive scripts in the stricter held-out audit; prevalence ≤12% is an upper bound).
- **Feature ablations**: stronger now. `reproducibility/real_corpus_ablation.md:11-25` shows the 5-feature ladder is **flat** on the 10-bug real corpus (baseline **5/8** cumulative, unchanged when disabling CEGAR/device/phase/grad/low-conf). `reproducibility/per_rule_ablation_60bug.md:18-41` gives the per-handler-family deltas on the 60-bug corpus.
- **McNemar / per-bug tables**: the headline numbers are present and consistent in `neurips.txt:14-18` and `965-987`, but the unfiltered post-freeze sample is explicitly reported as **directional** not significant (`5/15` vs FT `2/15`, Pytea `3/15`; `review_response.md:158-171`, `real_bugs_unfiltered.md:54-83`).

3) Strong positives
- Reproducibility is unusually strong: many claims have paired `.py` + `.json` + `.md` artifacts (`cv_caller_rely`, `lean_precondition_boundary_test`, `per_rule_ablation_60bug`, `real_corpus_ablation`, `real_bugs_unfiltered`).
- Frozen-corpus discipline is explicit (`real_bugs_unfiltered.md:14-18, 28-30`), with `postfreeze_catalogue_hash.txt` cited.
- The repo now distinguishes calibration vs end-to-end evidence for Dynamo and distinguishes headline catch-rate vs off-axis fires (`real_bugs_unfiltered.md:56-89`).● Read (Explore agent — Checking repo evidence)
  └ Completed

I’ve got both syntheses; I’m spot-checking the highest-impact claims before fixing the final score and wording.

● Read cv_caller_rely.md
  │ reproducibility/cv_caller_rely.md
  └ L1:60 (60 lines read)

● Read Extended.lean
  │ lean/TensorGuard/Extended.lean
  └ L85:108 (24 lines read)

● Read per_rule_ablation_60bug.md
  │ reproducibility/per_rule_ablation_60bug.md
  └ L1:60 (59 lines read)

● Read real_corpus_ablation.md
  │ reproducibility/real_corpus_ablation.md
  └ L1:40 (31 lines read)

● Read lake_build.md
  │ reproducibility/lake_build.md
  └ L1:40 (37 lines read)

● Read neurips.txt
  └ L680:730 (51 lines read)

● Read real_bugs_unfiltered.md
  │ reproducibility/real_bugs_unfiltered.md
  └ L50:95 (46 lines read)

● Read neurips.txt
  └ L1998:2035 (38 lines read)

● Read neurips.txt
  └ L950:1010 (60 lines read)

● Read neurips.txt
  └ L1:140 (140 lines read)

One more targeted read: I want the exact handler-scope split behind the mechanization claim, since that’s likely score-relevant.

● Read lean_precondition_boundary_test.md
  │ reproducibility/lean_precondition_boundary_test.md
  └ L1:35 (35 lines read)

● Read handler_scope_per_block.md
  │ reproducibility/handler_scope_per_block.md
  └ L1:55 (46 lines read)

## Summary
This paper presents TensorGuard, a no-execution static verifier for PyTorch `nn.Module` `forward` methods that tracks symbolic tensor shapes and first-order gradient-flow flags from class source without instantiation or tracing. The technical core is a refinement-type calculus with an assume/guarantee discipline at module boundaries, plus a partially mechanized Lean audit of the operator-rule table. Empirically, the paper reports 53/60 `REFUTED-PROOF` on a curated historical bug corpus, a 32/34 vs. 22/34 fragment-fair comparison against Pytea, and a pre-registered 5/15 result on a real-PR sample that it now describes as directional rather than statistically separable. The paper also argues for a one-way correspondence between TensorGuard’s refinements and TorchDynamo metadata guards, but scopes this as a necessary-direction, preliminary result rather than a full equivalence theorem. Overall, the paper tackles an important verification problem and is notably more calibrated than many systems papers, but the strongest claims remain limited by narrow real-world wins and by a mismatch between some paper-level mechanization statements and the current repository state.

## Prior weakness disposition
- [PARTIAL] **Headline real-world bug-finding result is weak.** On the only sample drawn without selection for fragment fit (Table 3, `N=15`)... -- The paper now explicitly labels the 5/15 result as directional and non-significant, but the headline real-world bug-finding evidence is still modest.
- [PARTIAL] **The Lean mechanization claim does not extend to soundness of the deployed verifier.** Only 28/79 handlers are Lean-audited... -- The text now narrows the claim and puts the analyzer in the TCB, but the theorem-backed footprint is still small and the repo still contains a live `sorry` contradicting the “11/11 ... closed sorry-free” wording.
- [RESOLVED] **Theorem 5 (Dynamo-guard correspondence) is over-scoped relative to its evidence.** It is stated as a statement over... -- The current paper scopes this as a necessary-direction, preliminary lemma with limited audited coverage rather than a broad equivalence claim.
- [PARTIAL] **The 128 ContractViolation verdicts depend on a synthesised caller-rely envelope whose realisability is checked only...** -- The paper adds a more careful bucket analysis, but the main text still leans on a single-default witness check and remains internally awkward next to the repo’s stronger “0/128 unwitnessed” artifact.
- [UNRESOLVED] **The first-order grad-flag lattice is admitted to be silently incorrect on parameter-sharing-under-renamed-attribute...** -- This remains an admitted limitation rather than a resolved issue, and the paper still reports only a prevalence bound.
- [PARTIAL] **Two of the three "discriminative" features in the per-feature stress benchmark (Table 5) are admitted to be no-ops...** -- The contribution claim is narrowed by dropping CEGAR and phase as contributions, but the repo’s real-corpus ablation still suggests even the remaining feature ladder is flat on the 10-bug upstream-faithful corpus.
- [UNRESOLVED] **Presentation.** The paper packs caveats into running prose to such a density that the actual claims become hard to extract... -- The current draft is still caveat-dense, especially in the Abstract, Contributions, and evaluation discussion.
- [UNRESOLVED] **The "32/34 vs. 22/34" McNemar result reports `b=10, c=0`** ... -- The paper still gives the McNemar result without an explicit per-bug contingency table that would let readers audit the matched-pair calculation.

## Strengths
- The problem is important and technically interesting: static checking of PyTorch shapes and gradient flow without instantiation or tracing would fill a real gap left by execution-based tooling.
- The paper is unusually calibrated in several places: it cleanly separates `REFUTED-PROOF`, `CONTRACT-VIOLATION`, `LIBRARY-WARN`, and abstentions, and it now openly states that the 5/15 real-PR result is not statistically separable.
- The evaluation is broader than a single curated corpus, spanning a historical bug corpus, a fragment-fair Pytea comparison, real-source blocks, real public-repo bugs, and several ablations and stress tests.
- The reproducibility package is strong: many claims are backed by dedicated scripts and artifacts rather than hand-written numbers.
- Narrowing Theorem 5 to a necessary-direction statement is the right move and makes the paper more believable than an over-ambitious equivalence claim would have.

## Weaknesses
- The paper’s mechanization story is currently inconsistent with the repository. The Abstract and Contribution C6 claim “11/11 soundness lemmas closed sorry-free,” but `lean/TensorGuard/Extended.lean` still contains a live `sorry`, and `reproducibility/lake_build.md` explicitly acknowledges it. That is a serious soundness-presentation mismatch.
- Even after the wording improvements, the theorem-backed audited footprint remains narrow relative to the deployed verifier. `reproducibility/handler_scope_per_block.md` reports that only 11/57 `Verified` blocks and 25/128 `ContractViolation` blocks stay entirely within Lean-or-pen-and-paper handlers; many in-scope verdicts still touch tested-only handlers.
- The real-world bug-finding story is still weak for a NeurIPS systems paper. The pre-registered real-PR result is 5/15 vs. FakeTensorMode 2/15 and Pytea 3/15, and the 488-block real-source corpus yields 0 unconditional `REFUTED-PROOF`, so the strongest empirical wins remain on curated or fragment-controlled settings.
- The `ContractViolation` evidence is not cleanly presented. Section 4.1 still foregrounds the 118/128 single-default witness check with 10 non-witnessed rows, while `reproducibility/cv_caller_rely.md` argues separately that there are 0/128 unwitnessed CVs under a broader constructor-pattern analysis; this discrepancy needs reconciliation.
- Contribution C5 remains hard to interpret. The paper says that on the real corpora only three knobs move verdicts, but `reproducibility/real_corpus_ablation.md` shows a flat five-feature ladder on the 10-bug upstream-faithful real corpus, which makes the practical importance of these auxiliary features unclear.
- The Pytea comparison is still not fully auditable from the paper alone. The 32/34 vs. 22/34 result and McNemar test are potentially meaningful, but without an explicit per-bug agreement table the reader cannot verify the matched-pair structure behind `b=10, c=0`.
- The paper is still overly dense with caveats and scope conditions. In particular, the Abstract and Contributions section interleave headline claims with multiple exceptions, which makes it difficult to tell what the central take-away actually is.

## Questions
- Can the authors reconcile the “11/11 soundness lemmas closed sorry-free” claim with the live `sorry` still present in `lean/TensorGuard/Extended.lean` and explain exactly what part of the mechanization should be considered complete?
- For theorem-backed verdicts, what are the paper-level headline numbers if one restricts to blocks whose full forward path stays entirely inside Lean-audited or pen-and-paper handlers?
- Which `ContractViolation` account should readers treat as authoritative: the paper’s 118/128 single-default witness check with 10 non-witnessed rows, or the repo artifact claiming 0/128 unwitnessed CVs under constructor-pattern analysis?
- Please provide the explicit 34-case contingency table for the TensorGuard vs. Pytea modern-subset comparison so the McNemar calculation can be directly audited.
- When Contribution C5 says that on the real corpora only three knobs move verdicts, which corpora does that statement refer to exactly, and how should it be reconciled with the flat `real_corpus_ablation.md` result on the 10-bug upstream-faithful corpus?

## Scores
Soundness: 2
Presentation: 2
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would move me up by one point is a fully consistent and defensible soundness/mechanization story: remove the live `sorry`, align the paper’s Lean claims with the actual audited footprint, and reconcile the paper/repo discrepancies around theorem-backed coverage and `ContractViolation` evidence. Right now those inconsistencies keep me from treating the strongest claims as fully supported.


Changes   +0 -0
Requests  1 Premium (2m 42s)
Tokens    ↑ 540.6k • ↓ 14.4k • 452.5k (cached) • 5.2k (reasoning)
