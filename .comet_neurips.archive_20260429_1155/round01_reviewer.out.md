● Read round01_reviewer.md
  │ .comet_neurips/round01_reviewer.md
  └ 72 lines read

● List repo (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls && echo "---" && ls .comet_neurips/
  └ 17 lines...

● Check meta info (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l neurips.tex neurips.txt README.md && cat
  │ .comet_neurips/history.jsonl 2>/dev/null | head -50 && echo "---" && cat
  │ .comet_neurips/scores.jsonl 2>/dev/null
  └ 12 lines...

● Read neurips.tex
  └ 172 lines read

● List sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls sections_v5/ && echo "---" && wc -l sections_v5/*.tex
  └ 2 lines...

● Find sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && find . -name "intro_v6*" -o -name "eval_v6*" -o -name "calculus_v6*"
  │ 2>/dev/null | head -20
  └ 4 lines...

● Section sizes (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l docs/paper/sections_v5/*.tex && echo "---" && ls src/ tests/
  │ benchmarks/ 2>/dev/null | head -100
  └ 132 lines...

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ 113 lines read

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ 162 lines read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:200 (200 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L200:450 (251 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L450:700 (251 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L700:872 (173 lines read)

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ 145 lines read

● Read related_v6.tex
  │ docs/paper/sections_v5/related_v6.tex
  └ 53 lines read

● Read impl_v6.tex
  │ docs/paper/sections_v5/impl_v6.tex
  └ 87 lines read

● Check artifacts (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/ 2>/dev/null && echo "---" && ls lean/TensorGuard/ 2>/dev/null &&
  │ echo "---" && ls reproducibility/ 2>/dev/null | head -40 && echo "---" && ls
  │ experiments_v5/v8/ 2>/dev/null | head -30
  └ 77 lines...

● Check lean sorries (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -l "sorry" lean/TensorGuard/*.lean 2>/dev/null && echo "---sorry
  │ counts---" && grep -c "sorry" lean/TensorGuard/*.lean 2>/dev/null && echo
  │ "---" && cat reproducibility/lake_build.log 2>/dev/null | tail -10
  └ 12 lines...

● Find sorries (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "sorry" lean/TensorGuard/V5OperatorRules.lean
  │ lean/TensorGuard/Extended.lean lean/TensorGuard/Parity.lean
  └ 7 lines...

● Check headline results (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python -c "import json;
  │ d=json.load(open('benchmarks/headline_results.json')); print(json.dumps(d,
  │ indent=2)[:2000])" 2>&1 | head -80
  └ 2 lines...

● Check headlines and unfiltered (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -c 2500 benchmarks/headline_results.json && echo "" && echo "---" &&
  │ head -c 1500 reproducibility/dynamo_falsification_audit.md 2>/dev/null &&
  │ echo "---" && ls experiments_v5/v8/real_bugs_unfiltered/ 2>/dev/null | head
  │ -25
  └ 78 lines...

● Check corpus json (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -c 500 benchmarks/blocks_corpus.json && echo "---" && head -c 1500
  │ benchmarks/real_benchmark_results.json && echo "---" && ls
  │ benchmarks/_corpus 2>/dev/null | head && wc -l benchmarks/blocks_corpus.json
  │ benchmarks/real_bug_corpus.json 2>/dev/null
  └ 74 lines...

● Find 488 references (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -rl "488" benchmarks/ reproducibility/ experiments_v5/ 2>/dev/null |
  │ head -10 && echo "---" && find . -path ./.git -prune -o -name "*.json"
  │ -print 2>/dev/null | xargs grep -l '"total_blocks"\|"n.*488"\|"N.*488"'
  │ 2>/dev/null | head -5
  └ 1 line...

● Read shell output Waiting up to 60 seconds for command output
  └ 1 line...

● Stop shell
  └ 1 line...

● Search (grep)
  │ "488" (benchmarks, reproducibility)
  └ 169 files found

● Check 488 corpus json (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -c 1500 reproducibility/no_assume_subset_488.json && echo "" && echo
  │ "---" && head -c 1500 reproducibility/lw_rp_gap.md
  └ 81 lines...

## Summary

The paper presents \textsc{TensorGuard}, a no-execution refinement-type checker for PyTorch \texttt{nn.Module} classes. Tensor types $\mathsf{Tensor}\{s,g\mid\varphi\}$ carry a symbolic shape, a flat grad-flag lattice, and a Z3-decidable side condition; assume/guarantee contracts compose at the class boundary; a backward verifier flags three silent-zero-grad bug classes; and a Dynamo-guard "necessary direction" inclusion lemma is claimed (Thm. 5). Empirically the authors report a 488-block real-source corpus on which TG returns 57 Verified / 0 unconditional Refuted-Proof / 128 Contract-Violation / 78 Library-Warn / 225 Abstain (vs. ≥435/488 N/A for the four PyTorch baselines and Pytea), a 60-bug historical corpus (53/60 RP), 10 upstream-faithful re-extracts (7/10 RP@0.99 + 1/10 @0.80), a held-out unfiltered N=15 post-freeze sample (5/15 catches vs. FakeTensorMode 2/15, Pytea 3/15), and a Lean 4 audit that closes 11/11 previously-axiomatic soundness lemmas sorry-free for 28 of 79 shape-transfer rules and agrees with torch 2.9.1 on 28 000/28 000 in-fragment cases.

## Strengths

- The reporting is unusually calibrated for this style of paper: the headline triple is split into RP / CV / LW; only RP and CV are claimed to be covered by Thm. 1; LW and Abstain make no soundness claim; and the user-visible (free-symbolic-config) recomputation 34V/0RP/206LW/248A is reported alongside the assume-envelope number.
- The Lean artefact is real and non-trivial: \texttt{lean/TensorGuard/} builds sorry-free under \texttt{lake build} (verified: 0 \texttt{sorry} occurrences in proofs across \texttt{Soundness.lean}, \texttt{AssumeGuarantee.lean}, \texttt{V5OperatorRules.lean}; the textual \texttt{sorry} hits in those files are inside comments). The 28 000-sample byte-mirror differential against torch 2.9.1 plus the off-envelope boundary check on ~2 400 mutators is the right shape of evidence.
- The post-freeze protocol (catalogue freeze \texttt{040f6f3} on 2026-04-07, pre-registered query 2026-04-08) and the explicit retention of out-of-fragment cases as honest abstentions in the N=15 unfiltered sample is a methodologically clean way to bound the fragment-fit confound.
- The handler-soundness scope table (Tab. \ref{tab:handler-soundness}) and the per-block scope accounting (\texttt{reproducibility/handler\_scope\_per\_block.md}) explicitly partition verdicts by which audit class their handlers fall into — a level of bookkeeping most refinement-type tool papers skip.

## Weaknesses

- **The paper's central headline is structurally fragile.** On the 488-block real-source corpus the verifier returns 0 unconditional Refuted-Proof verdicts, and the user-visible (no synthesised \texttt{config} envelope) recomputation collapses Verified from 57 to 34 and CV+RP to 0/0. The "find real bugs in real source" claim is therefore carried entirely by the 60 historical bugs (selected for shape-error keywords), the 10 upstream-faithful re-extracts, and the N=15 post-freeze sample. The 488-block evaluation is, on its own terms, a coverage-and-abstention measurement rather than a bug-finding result, and the abstract should not be allowed to read as if the 488 number is doing bug-finding work.
- **The "TG ≫ baselines" reading on the 488-block corpus is corpus-baked.** ≥435/488 N/A for FakeTensorMode/torch.fx/torch.export/torch.export/Pytea is a structural fact about constructor-argument requirements on a corpus that was selected to consist of \texttt{nn.Module} classes that need a non-trivial \texttt{config}. The paper acknowledges this once (line 200–204 of \texttt{eval\_v6.tex}) and then continues to use the 488-block triple as the headline framing in the abstract and conclusion. Either the 488 number is fairness-compatible with the N/A baselines (it isn't) or it should be demoted to "applicability" rather than presented next to the bug-finding numbers.
- **The fair head-to-head N=34 vs. Pytea is a constructed subset.** The "modern subset" is defined by the static AST predicate "every operator called from \texttt{forward} appears in Pytea's 2022 \texttt{pylib/}". Since TG is restricted symmetrically (Sec. \ref{sec:eval-benchmark}, "Pytea modern-subset filter" paragraph), the comparison is internally consistent, but the selection still excludes precisely the bugs Pytea cannot reach by construction. Please report (i) the size of the symmetric handler intersection vs. each tool's full catalogue, and (ii) the corresponding TG-only / Pytea-only refute counts on the *complement* (the 26 bugs that fall out of the modern subset) so the reader can judge whether the 32/34 vs 22/34 gap is a fragment-coverage artefact.
- **The post-freeze N=15 result is presented as the deciding generalisation evidence but the head-to-head is not statistically separable.** TG 5/15 vs. FakeTensorMode 2/15 (Fisher-exact two-sided p=0.39) and vs. Pytea 3/15 (p=0.68). The paper concedes this once and then in the abstract still asserts TG is "point-strictly above" the baselines; on N=15 a difference of 2–3 catches is well inside binomial noise and should not be in the abstract as a separation result. Please either grow the unfiltered sample to a size where Wilson intervals on the rate difference exclude zero, or remove the comparative claim from the abstract.
- **Thm. 5 (Dynamo-guard correspondence, necessary direction) is close to vacuous as stated.** "Every shape/dtype/rank refinement bit Dynamo reads on the trace at any input $x$ with $A(x)$ true is a refinement variable in $\varphi$ for some rule in the catalogue" is *true by construction* whenever the catalogue is the set of metadata bits TG inspects, because TG's catalogue is also defined off Dynamo's specialiser bits. The empirical falsifier (Sec. \ref{sec:eval-dynamo}, "Falsification predicate" paragraph) reports {SHAPE:0, DTYPE:0, RANK:0, INT:48} — i.e., the 48 in-contract recompiles all fall in the bucket the theorem already excludes. A non-vacuous test would be: a catalogue-extension experiment where you intentionally widen TG's refinement language to include one Dynamo guard kind not currently in the catalogue (e.g., one of the integer or list-length specialisers), measure whether a recompile in that kind has ever fallen on an in-contract input, and report whether the theorem still holds at the wider catalogue. The current audit cannot distinguish "Dynamo guards are a superset of TG refinements" (a tautology) from "Dynamo's shape/dtype/rank guards are exactly TG's refinements" (the substantive claim).
- **The Lean audit covers the rule table, not the analyser.** This is stated honestly in §\ref{sec:eval-lean}, but the consequence for the reader is sharper than the paper admits: only 28/79 shape handlers are Lean-audited, the AST extractor and Python analyser are not, and the assume/guarantee composition theorem (Thm. 3) is mechanised on a 3-operator DSL only. Per the per-block scope accounting (line 845–852 of \texttt{eval\_v6.tex}), only 36/185 in-soundness verdicts on the 488-block corpus touch only Lean-audited or pen-and-paper handlers — meaning the soundness theorem applies tightly to roughly one-fifth of the headline verdicts. The "sound static verification" framing in the title is therefore stronger than the underlying mechanisation.
- **Two of the per-feature ablation knobs are conceded to be no-ops in the implementation.** Sec. \ref{sec:eval-loc-hybrid} reports CEGAR contract discovery and the train/eval phase check as "honest no-ops" on real data; CEGAR predicates are stored as metadata but never surfaced as \texttt{Bug} objects. These are listed alongside the active knobs in Tab. \ref{tab:ablation} and in the contributions list, even after the abstract concedes the empirical contribution is now scoped to three features (device, grad, low-conf gating). Please remove the inactive features from the implementation/contribution claims rather than carrying them as engineering "ships".
- **The leave-one-category-out result (53/60 → 53/60) is a non-result that the paper presents as robustness.** When LOO leaves the rate unchanged across every category drop, the most parsimonious explanation is that the LOO is hitting orchestration code rather than the load-bearing rules. The paper acknowledges this and pivots to a "true rule-class LOO" that *also* yields 53/60 → 53/60, attributing the constancy to an independent AST-pattern path. If catches survive simultaneous removal of (a) the per-category operator handlers and (b) the AST-pattern intent-bug analyser, then a substantial fraction of the 53/60 is attributable to a constraint-based shape backend that is not what the paper's calculus and Lean audit are about. Please report the catch count when *all three* paths (per-operator handlers, intent-bug AST patterns, constraint-based backend) are disabled, and the catch count of the constraint backend alone, so the reader can attribute the 53/60 to a specific component.
- **Several rb_* "catches" were engineered into the system inside the development window.** Lines 172–192 of \texttt{eval\_v6.tex} note that the rb_003/rb_004/rb_006/rb_010 conversions from silent-verified to RP@0.99 came from a "three-stage constructor-bound integer-attribute envelope synthesiser" and a "per-forward local-scalar map" added within the round-4 to round-6 window. The pre-registered post-freeze split (N=6 + N=15) is the only segment of the real-bug evaluation that is uncontaminated by handler additions designed against specific bugs; on that split the catch rate is 3/6 + 5/15. The headline 7/10 should be reported with a clear footnote that ≥4 of the 7 catches were enabled by handler edits made after the bugs were inspected.

## Questions

- On the 488-block corpus, what is the user-visible (no-assume) catch rate on a *bug-injected* subset — i.e., if you mutate one shape-arithmetic axis in each of, say, 100 of the 488 blocks, how many does TG flag as RP (not CV) without any synthesised \texttt{config} envelope? This isolates the calculus's bug-finding power from the catalogue-coverage and assume-synthesis layers.
- For the N=15 unfiltered post-freeze sample, please report the per-PR overlap matrix (which PRs each tool catches), not just the marginal counts. With N=15 and 5/2/3 catches the joint distribution is the only way to tell whether TG's catches *contain* the FakeTensor/Pytea catches or are disjoint from them.
- Can you provide a falsifier for Thm. 5 that does not reduce to the catalogue's own definition? E.g., an audit where TG's refinement language is *extended* by one dtype-promotion guard kind currently outside the catalogue, and the in-contract recompile bucket for that kind is reported on the same 17-module audit.
- The backward verifier's $500/500$ static↔runtime agreement and 8/8 canonical-bug catch are reported on synthetic small modules and 8 hand-curated cases respectively; the 10-real-model sweep explicitly excludes \texttt{torch.utils.checkpoint} and parameter sharing. What is the agreement rate on a sample of the (≤12% prevalence) training scripts that *do* use these constructs?
- On the joint LOO that disables per-category handlers and the AST-pattern intent-bug analyser simultaneously, you still report 53/60. What is the catch count when you additionally disable the constraint-based shape back-end (the "third path")? If it is 53/60 still, what catches the bugs?
- For the modern-subset Pytea comparison, what is the size of the *complement* (60-block bugs that fall outside the symmetric handler intersection) and the per-tool catch counts on it?

## Scores

Soundness: 2
Presentation: 3
Contribution: 2
Confidence: 4
Overall: 4

## Borderline reasons

The single change that would push the overall score from 4 to 5 is to grow the pre-registered, post-freeze unfiltered real-PR sample from N=15 to N≥40 with the same protocol, so that the head-to-head Fisher-exact comparison against \texttt{FakeTensorMode} and Pytea can either separate at α=0.05 or be honestly reported as null. As it stands, the unfiltered post-freeze sample is the only bug-finding evaluation segment that is uncontaminated by either corpus-baked baseline-inapplicability (488 blocks) or in-development handler edits (rb_001–rb_010), and at N=15 it cannot carry the comparative-bug-finding headline the abstract leans on.


Changes   +0 -0
Requests  7.5 Premium (5m 40s)
Tokens    ↑ 1.8m • ↓ 10.3k • 1.7m (cached)
