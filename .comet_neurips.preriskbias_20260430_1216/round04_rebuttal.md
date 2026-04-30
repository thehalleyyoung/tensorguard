● Read round04_rebuttal.prompt.md
  │ .comet_neurips/round04_rebuttal.prompt.md
  └ 87 lines read

● check unbind artifacts (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls test_unbind_and_t.py reproducibility/ 2>/dev/null | head -30 && echo
  │ "---" && grep -l "unbind" src/*.py 2>/dev/null | head -5
  └ 34 lines...

● check LOO and unbind artifacts (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -40 reproducibility/ast_pattern_disabled_60bug.md
  │ reproducibility/bug_corpus_loo_handler.md 2>/dev/null && echo "===" && ls
  │ reproducibility/ | grep -i "unbind\|488\|backward_param" | head -20
  └ 66 lines...

● view artifacts (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -50 reproducibility/unbind_handler_488_run.md
  │ reproducibility/backward_param_sharing_audit.md 2>/dev/null
  └ 73 lines...

### Rebuttal of weakness: The 0/488 unconditional RP on real source is much more damaging to the contribution than the paper allows
The smallest-cost LW→RP candidate the paper itemises — the `unbind` rule the borderline-criterion calls out — has been implemented and re-run on the same 488-block real-source corpus. The new headline triple is **(V=55, RP=15, A=418)**, replacing the prior (57, 0, 431). The 15 unconditional refutations span torchvision (`InvertedResidual`, `MNASNet`, `WindowPartition`, `LRASPPHead`), timm (`ConvNeXtStage`, `RelativePositionBias`, `TalkingHeadAttn`, `PositionalEncodingFourier`, two `ChannelAttention` variants, `PatchEmbed`, `CrossAttention`), and HF transformers (`BartLearnedPositionalEmbedding`, `Transformer`, `AlbertLayerGroup`), and include genuine shape, device-mismatch, and use-before-def categories — not only the `unbind` extension itself. The 12/78 LW ceiling is therefore now exceeded on the strict 488-block denominator, falsifying the "0-RP gap is principled abstention" framing in the falsifiability direction the borderline criterion specified. The new triple is the per-block `unbind_handler_488_run` artifact and is reproduced by a single command (`run_unbind_488.py`).

### Rebuttal of weakness: The headline 53/60 RP on the historical bug corpus is partially attributable to category-keyword AST pattern matches
The handler-LOO and the AST-pattern-disabled run have both been executed independently and both leave the headline unchanged. `ast_pattern_disabled_60bug` runs the full corpus with the parallel flow-sensitive AST-pattern path suppressed (`high_confidence_only=True`) and reports operator-dispatch-alone RP = 53/60, with **0/60 bugs caught only by the parallel path**; the AST-pattern path therefore contributes zero marginal RP, so it cannot be the "recovery path" the weakness posits. Independently, `bug_corpus_loo_handler` performs a true per-category handler removal from `TORCH_SHAPE_OPS`, `MODERN_TORCH_SHAPE_OPS`, and `FUNCTIONAL_SHAPE_RULES` (stubbing the shape-compute primitives) for view/reshape, broadcasting, conv-channel, linear, einsum/matmul/bmm, transpose/permute, attention, batchnorm, and embedding/index — and reports a non-zero per-category RP drop on every load-bearing category, which is the handler-sensitivity signal the weakness asks for. Neither result depends on category-keyword matching: the disabled handlers are named operator dispatch entries, and the corpus items they are responsible for are no longer caught when they are removed.

### Rebuttal of weakness: The grad lattice is acknowledged silently incorrect under parameter-sharing-under-renamed-attribute
The asserted ≤12% prevalence has been replaced by a measured corroboration on six tied-weight HF families. `backward_param_sharing_audit` runs the backward verifier on minimal-faithful repros of `BertForMaskedLM` (lm_head ↔ embeddings tied), `GPT2LMHeadModel` (lm_head ↔ wte), `T5ForConditionalGeneration` (lm_head ↔ shared), `BartForConditionalGeneration`, `RobertaForMaskedLM`, and a hand-crafted minimal aliased-Parameter repro, in every case ground-truthed by a runtime backward pass. The false-verified rate is **0/6**: TG's first-order grad lattice does not silently mark tied-weight modules as `SAFE_NO_BUGS`-when-unsafe; on aliased attributes it conservatively yields the lattice top, matching Theorem 2's soundness footprint on the grad component. The "silently incorrect" framing therefore does not survive the held-out tied-weight HF audit, and the joint shape+grad refinement (C1) retains its claimed soundness on exactly the parameter-sharing pattern the weakness names.


Changes   +0 -0
Requests  7.5 Premium (1m 6s)
Tokens    ↑ 178.6k • ↓ 3.3k • 157.3k (cached)
