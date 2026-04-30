● List directory .comet_neurips
  └ 45 files found

● Read round11_reviewer.md
  │ .comet_neurips/round11_reviewer.md
  └ 120 lines read

I’m checking the paper against the repo evidence and the prior weaknesses so the review reflects the current artifact, not earlier rounds.

● Read round10_reviewer.out.md
  │ .comet_neurips/round10_reviewer.out.md
  └ 116 lines read

● Read README.md
  └ L1:260 (257 lines read)

● Search (grep)
  │ "488|371|Pytea|Lean|Theorem 1|soundness|Llama|McNemar|25/34|32/34|79
  │ handlers|17 of 79|single-default-omitted|0/25|13.32|60-bug|10-real-bug"
  │ (neurips.txt, README.md, docs)
  └ 8 lines found

I’ve got the prior review context; now I’m reading the exact paper sections tied to those seven weaknesses to see what actually changed.

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ L1:180 (135 lines read)

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ L1:220 (186 lines read)

● Read impl_v6.tex
  │ docs/paper/sections_v5/impl_v6.tex
  └ L1:160 (86 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:260 (260 lines read)

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ L1:220 (171 lines read)

● Search (grep)
  │ "53/60|60 bug|10 real|488|371|128
  │ CV|single-default-omitted|15/488|3\.07|32/34|25/34|McNemar|Pytea|Lean|28 of
  │ 79|17 of 79|79 handlers|Theorem
  │ 1|soundness|Llama|0/25|13\.32|GPT-NeoX|einsum|conv2d|backward
  │ verifier|tied/renamed" in *.tex (docs/paper/sections_v5)
  └ 7 lines found

I found several concrete updates: the paper now adds an AST-oracle audit for the CV path, gives an input-shape-contract rerun at 15/488, and still states 0 unconditional RP on the 488-block free-symbolic-config slice. I’m reading the remaining evaluation subsections to calibrate the new score.

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L980:1421 (366 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L260:620 (361 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L620:980 (361 lines read)

● Read review_response.md
  └ L1:240 (240 lines read)

The paper added a few nontrivial pieces since the prior round — especially modern baselines and a direct AST-oracle audit — so I’m spot-checking the repo artifacts behind those additions before I settle the dispositions.

● Read cv_caller_rely.md
  │ reproducibility/cv_caller_rely.md
  └ L1:220 (61 lines read)

● Read dynamo_e2e.md
  │ reproducibility/dynamo_e2e.md
  └ L1:220 (55 lines read)

● Read lean_precondition_boundary_test.md
  │ reproducibility/lean_precondition_boundary_test.md
  └ L1:220 (55 lines read)

● Search (grep)
  │ "34/34|jaxtyping|beartype|25/34|32/34|torch\.compile|dynamo_e2e|cv_caller_re
  │ ly|140/140|symbolic_config_attrs|0/25|13\.32|25\.0%|LlamaAttention|LlamaMLP_
  │ buggy|17/79|28 of 79|applyOpExt_sound_unbind|lake build|sorry-free"
  │ (reproducibility, experiments_v5, lean)
  └ 6 lines found

## Summary
This paper presents TensorGuard, a static verifier for PyTorch `nn.Module` class source that reasons about tensor shapes and gradient-flow flags without instantiating the model or executing a trace. The technical core is a refinement-typed calculus with Z3-dischargeable obligations plus an assume/guarantee discipline for module composition, with partial Lean 4 mechanisation of operator rules and a 17-operator composition fragment. Empirically, the paper reports 53/60 Refuted-Proof results on a historical bug corpus, 7/10 on upstream-faithful real public bugs, and 5/15 catches on an unfiltered post-freeze real-PR sample, while explicitly acknowledging 0 unconditional RP on the 488-block free-symbolic-config real-source corpus. The evaluation also now includes a fragment-fair Pytea comparison, contemporary execution-based baselines on the same 34 bugs, and audits aimed at bounding the trusted computing base. Overall, the paper is careful and unusually candid about scope, but the strongest practical-evidence claims still come from relatively small bug-centric datasets rather than the headline real-source corpus.

## Prior weakness disposition
- [UNRESOLVED] The headline empirical claim of the paper is carried almost entirely by the 60-bug + 10-real-bug + 6+15 post-freeze corpora... -- The paper still reports 0/488 unconditional RP on the free-symbolic-config 488-block corpus, so the practical utility claim remains driven by much smaller bug-focused datasets.
- [RESOLVED] The fragment-fair Pytea head-to-head (32/34 vs. 25/34, McNemar p=0.0156) compares against a tool whose upstream... -- Section 4.1 now adds contemporary baselines on the same 34 bugs, including `torch.compile` at 34/34 and jaxtyping+beartype at 0/34, so the stale-Pytea comparison is no longer the only comparator carrying the evaluation.
- [PARTIAL] The Lean audit covers 28 of 79 handlers and the AG composition theorem covers 17 of 79 operators, with the parser... -- The paper adds boundary testing and targeted load-bearing mutation results, but the mechanised scope and analyser-wide 7/50 union mutation-kill rate remain limited.
- [PARTIAL] Theorem 1 (Soundness) covers RP and CV verdicts, but in the user-visible regime there are 0 RP and 128 CV verdicts... -- The new AST-oracle and caller-rely audits materially strengthen the CV story, but the 488-block soundness case still depends on unaudited TCB components rather than end-to-end mechanisation.
- [UNRESOLVED] The cross-family Llama 2/3 sanity result (4/6 V, 2/6 RP including a buggy variant) is worth keeping but is six... -- This remains a six-module sanity check with one deliberate bug fixture, so it is still too small to support strong cross-family generalisation claims.
- [UNRESOLVED] The stub-mocked runtime sample on the 371-Verified subset (0/25 silently incorrect) is a Wilson upper bound... -- The paper still reports a small self-selecting 0/25 audit alongside 2/8 worst-case alias-family misses, so this remains an audit rather than a strong prevalence guarantee.
- [UNRESOLVED] The worked GPT-NeoX symbolic-calculus example is helpful as an illustration but a single end-to-end SMT trace... -- The body still offers essentially one fully worked symbolic proof path, so the calculus remains under-illustrated on other important operator families.

## Strengths
- The paper is exceptionally well calibrated about what is and is not claimed: the five-way verdict taxonomy, the explicit separation of RP/CV/LW, and the repeated disclosure of 0 unconditional RP on the free-symbolic-config 488-block corpus are all to its credit.
- The technical package is substantive: a refinement calculus, assume/guarantee composition, partial Lean mechanisation, a backward verifier, and a broad reproducibility surface.
- The new AST-extractor oracle audit and CV caller-rely audit directly improve one of the most important prior concerns, namely whether the many CV verdicts on the 488-block corpus rest on an opaque synthesis pipeline.
- The addition of contemporary execution-based baselines materially improves the empirical positioning: the reader can now see both where TensorGuard loses (`torch.compile` on executable repros) and where its class-source/no-instantiation regime is genuinely different.

## Weaknesses
- The paper's practical utility on natural-distribution real library source remains weak. Table 1 and Section 4.1 still show 0/488 unconditional RP on the user-visible free-symbolic-config regime, and even the strengthened rerun reaches only 15/488 once input-shape contracts are supplied.
- The strongest contemporary baseline on the fragment-fair executable subset is actually `torch.compile`/FakeTensor at 34/34, above TensorGuard's 32/34. This does not invalidate the paper's class-source setting, but it means the empirical story is about a different operating regime, not about outperforming the best available executable checker when execution is possible.
- The formal-assurance surface is still narrow relative to the user-visible claims. Section 4.4 reports only 36/185 V/CV verdicts on the 488-block corpus as touching only Lean-audited or pen-and-paper handlers; most verdicts still rely on tested-only handlers and TCB code.
- The robustness story is improved but still not strong at the analyser level: the best-of-union mutation-kill rate remains 7/50, and the stronger conv2d/einsum numbers come from targeted, load-bearing handler experiments rather than a broad system-wide audit.
- The cross-family evidence remains thin. The Llama 2/3 result is six modules with one synthetic bug fixture, and one of the two RP outcomes is a division guard on `num_heads`, not a caught real bug.
- The backward-verifier deployment story is still bounded by small, selective audits: 0/25 on a subset of 371 Verified rows, 1/42 held-out HF training scripts, and 2/8 false-verified on the worst-case alias/checkpoint family are useful caveats, but not yet a strong external validation.

## Questions
- Can the authors provide a larger natural-distribution bug benchmark in which class-source-only analysis is required and the bug labels come from external ground truth rather than curated historical repro corpora?
- In the 15/295 denominator for the strengthened rerun, how much of the excluded 193-row gap is due to the block-extraction protocol versus genuine fragment limitations of TensorGuard?
- Theorem 1's statement in Section 3 explicitly includes the `Verified` verdict, but Section 4 and the limitations text repeatedly say the soundness theorem covers RP and CV only. Which scope should the reader use when interpreting the 57 Verified verdicts on the 488-block corpus?
- What is the minimal alias-aware extension of the grad lattice that would catch the 2/8 renamed-attribute / parameter-sharing failures, and would that extension compose cleanly with the current assume/guarantee framework?

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
The clearest path to a higher score is a substantially stronger natural-distribution real-source result: either a non-trivial unconditional RP rate on the 488-block corpus or a larger externally labeled benchmark where the class-source/no-instantiation regime is truly necessary. Right now the paper is technically interesting and unusually honest, but its practical case is still carried mostly by curated bug corpora rather than by the headline real-source setting.


Changes   +0 -0
Requests  1 Premium (2m 4s)
Tokens    ↑ 244.7k • ↓ 7.5k • 179.2k (cached) • 4.1k (reasoning)
