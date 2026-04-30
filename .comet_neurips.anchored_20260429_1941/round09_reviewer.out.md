● Read round09_reviewer.md
  │ .comet_neurips/round09_reviewer.md
  └ 118 lines read

● List repo and check paper sources (shell)
  │ ls
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/ |
  │ head -50 && echo "---" && wc -l /Users/halleyyoung/Documents/div/mathdive
  │ rgence/halley-labs/tensorguard/neurips.tex 2>/dev/null && wc -l /Users/hal
  │ leyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/neurips.txt …
  └ 42 lines...

● Read neurips.tex
  └ 90 lines read

● List sections (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/s
  │ ections_v5/ 2>/dev/null && wc -l /Users/halleyyoung/Documents/div/mathdiverg
  │ ence/halley-labs/tensorguard/sections_v5/eval_v6.tex
  └ 1 line...

● Find section files (shell)
  │ find /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ -name "eval_v6.tex" -o -name "intro_v6.tex" -o -name "calculus_v6.tex"
  │ 2>/dev/null | head
  └ 7 lines...

● Section sizes (shell)
  │ wc -l /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguar
  │ d/docs/paper/sections_v5/*.tex
  └ 31 lines...

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ 1 line read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L180:320 (141 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L560:700 (141 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L900:1050 (151 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1100:1209 (110 lines read)

● Find post-freeze sections (shell)
  │ grep -n "post-freeze\|pf_005\|pf_001\|N=15\|FakeTensor"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -40
  └ 22 lines...

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L420:530 (111 lines read)

● Check C3 audit detail (shell)
  │ grep -n "examples/pytorch\|1/42\|silent-error-positive\|run_translation"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -20
  └ 4 lines...

● Find class-context discussion (shell)
  │ grep -n "199\|not-analysable\|class context\|class-context"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -20
  └ 4 lines...

## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module.forward` that statically verifies tensor shapes and gradient flow from class source. The paper reports `Refuted-Proof` on 53/60 (88.3%) of a curated bug corpus, a fragment-fair head-to-head against Pytea on N=34 (32 vs 22; McNemar exact p=0.00195), and a pre-registered post-freeze real-PR sample of N=15 where TG catches 5/15 vs. FakeTensorMode 2/15 and Pytea 3/15 (directional, not separable at α=0.05). The shape-transfer rule table is mechanised in Lean 4 (28 of 79 handlers, 11/11 soundness lemmas closed sorry-free), with the analyser implementation, AST extractor, backward verifier, and Z3 dispatch in the TCB. On the 488-block real-source corpus, the user-visible free-symbolic regime returns 0 unconditional `Refuted-Proof`; this is presented as a fragment-coverage measurement, not a bug-finding result.

## Prior weakness disposition
- [UNRESOLVED] The user-visible 488-block bug-finding headline is still 0 unconditional Refuted-Proof, and no new statistically-separable real-source result against either FakeTensorMode or Pytea -- The N=15 post-freeze sample (5/15 vs 2/15, 3/15) is still not separable at α=0.05 (Fisher p=0.39, 0.68); only a power calculation is added (eval_v6.tex:482-492), no fresh wave with separation.
- [UNRESOLVED] The 199/488 "not-analysable" Abstain bucket is still attributed to an extractor that strips class context, no per-class-context re-extraction rate -- The text (eval_v6.tex:195-213) only restates the diagnosis; no random-sample re-extraction with class context is reported.
- [PARTIAL] The Theorem-5 falsifier predicate is still vacuously satisfied -- The Track-E audit is extended from CNN-13 to 55 modules (eval_v6.tex:904-930) and the result (0 SHAPE/DTYPE/RANK guards across 72 in-contract recompiles) is reported, but it is reframed as a "denominator audit, not a falsifier evaluation"; the falsifier predicate remains vacuously satisfied on every non-curated population.
- [PARTIAL] The C3 held-out HuggingFace examples/pytorch audit reports 1/42 silent-error-positive but no analyser verdict on the 1 -- A new 8/8 trainer-harness measurement (eval_v6.tex:1160-1179) is added on a *different* checkpointing-enabled set, but the verdict on the actual 1/42 script identified by the AST-grep audit is still not reported.
- [UNRESOLVED] Rule-development holdout: per-bug random partition with unique catches per path -- Text (eval_v6.tex:584-613) reports each path alone refutes 53/60 and asserts "per-bug attribution overlaps almost everywhere"; no per-bug random partition with each path's unique-catch count is given, only aggregate keyword load-bearing shares.
- [UNRESOLVED] McNemar p=0.00195 still load-bearing; only out-of-corpus head-to-head (post-freeze N=15) is not separable -- Acknowledged in text (eval_v6.tex:476-492) with a power calculation, but no second wave executed; the load-bearing significance test remains the in-corpus N=34.

## Strengths
- The Lean 4 audit of the operator-rule table is real and tight: 28 rules with 11/11 lemmas sorry-free under `lake build`, plus byte-mirror diff (28,000/28,000 agree) and a non-trivial off-envelope boundary check (~2,400 samples, no silent-through). The handler-soundness scope table (28 Lean / 7 pen-and-paper / 44 tested-only) is honest about what is and is not mechanised.
- Calibration of the headline is admirably explicit: the abstract concedes 0 unconditional RP on the user-visible 488-block regime, distinguishes "fragment-coverage" from "bug-finding", and reports Wilson/Clopper-Pearson intervals throughout. The 25-block hybrid falsification (Table `tab:hybrid-falsify`) showing 20 TG-only and 5 FakeTensor-only catches credibly demonstrates complementary, not coincident, surfaces.
- Robustness measurement is multi-pronged: 50-mutation AST sweep on three corpora (best-of 7/50 = 14%) plus four hand-picked TCB faults bound the analyser from below and above. The TCB-exposure scan correctly treats the AST extractor / Z3 dispatch as untrusted.
- The post-freeze sample is genuinely pre-registered (query and inclusion frozen 2026-04-08, one day after catalogue freeze 2026-04-07), out-of-fragment cases retained as Abstain, and the off-axis fire on `rb_uf_010` is recorded as a false positive — not silently dropped.

## Weaknesses
- The headline statistical separability is still carried entirely by the in-corpus N=34 fragment-fair McNemar p=0.00195 (eval_v6.tex:280-292). Every out-of-corpus head-to-head fails to separate at α=0.05. A *single* additional pre-registered post-freeze wave at the power-calculated N_new=26 (TG vs. FakeTensorMode, eval_v6.tex:487-488) would either confirm or kill the separability claim; running it is the load-bearing missing experiment, not another power table.
- The 199/488 "not-analysable" Abstain bucket (eval_v6.tex:195-213) remains attributed to an extractor artefact, but the obvious falsification — re-extracting a uniform random sample of, say, 30 of the 199 rows *with* the surrounding `nn.Module` class context and reporting how many transition out of "not-analysable" — is still not run. Without that number, the claim that 199 is an extractor artefact rather than a fragment-coverage limit is unfalsified.
- The "over-determined catalogue" / rule-development-holdout claim (eval_v6.tex:584-613) — that the AST-pattern path *alone* and the operator-dispatch path *alone* each refute 53/60 — is striking and (if true) means at least one of the two paths catches every bug. The paper should report, for each of the 60 bugs, which paths catch it (a 60×2 table), so the reader can audit unique-catch counts and the "almost everywhere" overlap claim. As currently written it is impossible to tell whether the two paths are genuinely independent or whether one delegates to the other.
- The 55-module Track-E extension (eval_v6.tex:904-930) returns 0 SHAPE/DTYPE/RANK guards across 72 in-contract recompiles, all kind `Int`. Theorem 5's necessary direction is therefore tested only on the curated CNN-13 plus 5 hand-built `torch.library.custom_op` fixtures; on the population the theorem is meant to characterise, the falsifier predicate has *never* been exercised. Either run the falsifier on a population that *does* produce SHAPE/DTYPE/RANK guards (e.g. transformer modules with dynamic dims under a different recompile policy) or restrict the theorem's scope to "modules whose Dynamo recompiles are integer-specialisation-only".
- The C3 held-out audit (eval_v6.tex:1145-1158) identifies exactly 1/42 silent-error-positive script in `examples/pytorch/`. The paper should report TensorGuard's actual verdict (RP / Verified / Abstain) on that 1 script; the 8/8 trainer harness (eval_v6.tex:1160-1179) is on a different, hand-picked HF-head set and does not substitute. As stated this is a ground-truth count without an evaluation.
- The 25-row stub-mocked runtime sample on the 371-Verified subset (eval_v6.tex:1181-1209) — 0/25 silently-incorrect, Wilson 95% CI [0.00%, 13.32%] — is an honest start but the upper bound is wide. The paper would benefit from extending this to N=100 (Wilson upper would then be ≤3.6% under the same 0/N observation), which is mechanical given the harness already exists.
- The 488-block "two denominators" framing (eval_v6.tex:204-207) reports 14/488 = 2.87% RP under an *input-shape contract* and 14/289 = 4.84% on the analysable subset, but neither of these is the user-visible regime, and the abstract still cites the 0/488 number. Pick one denominator for the user-visible headline and report the other in an appendix; carrying three rates (0, 14/488, 14/289) in the body invites cherry-picking accusations.

## Questions
- Will you commit to a second pre-registered post-freeze wave at N_new=26 (TG vs FakeTensorMode) before camera-ready, and report the resulting Fisher exact p without further filtering?
- For the 199 not-analysable rows: on a uniform random subsample of, say, 30 rows re-extracted with surrounding class context, how many become analysable under the existing analyser?
- For the 60-bug corpus: please publish the per-bug × per-path matrix (AST-pattern path, operator-dispatch path, constraint-residue path). How many bugs are caught uniquely by each path?
- What verdict does TensorGuard return on the single silent-error-positive script identified by the AST-grep sweep over the 42 HF `examples/pytorch/` training scripts?
- Why does the 55-module Track-E audit produce 0 SHAPE/DTYPE/RANK guards under `dynamic=True`? Does this hold under `dynamic="auto"` or with `torch._dynamo.config.assume_static_by_default = False`? What is the smallest population on which the falsifier predicate is non-vacuous?
- The off-axis fire on `rb_uf_010` is the only false positive in the post-freeze sample. Are there analogous off-axis fires on the 488-block corpus or the 60-bug corpus that are currently being counted as RP catches?

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 3
Overall: 7

## Borderline reasons
A single fresh pre-registered post-freeze wave at N_new=26 against FakeTensorMode (already power-calculated in the paper) that produces a Fisher p<0.05 — combined with reporting TG's verdict on the 1/42 HF `examples/pytorch/` silent-error-positive script — would convert the load-bearing significance claim from in-corpus to out-of-corpus and remove the strongest remaining "not statistically separable" objection, pushing Overall to 8.


Changes   +0 -0
Requests  7.5 Premium (2m 30s)
Tokens    ↑ 683.3k • ↓ 7.5k • 644.3k (cached)
