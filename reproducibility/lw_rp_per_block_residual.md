# LW->RP Residual: Per-Block Enumeration

## Command

```
python3.11 reproducibility/lw_rp_per_block_residual.py
```

## Inputs

- 488-block real-source corpus, 78 LW verdicts.
- 12 are 'fragment-only' (forward body uses only catalogue ops).

## Result

- 12/12 predicted to convert to RP under the named single missing rule.
- Roadmap operator-rule additions: 5/12.
- Fragment-extensions outside the operator-rule discipline: 6/12.
- Mixed (one of each): 1/12.

## Per-block table

| block id | missing rule | extension class | witness shape | predicted verdict |
|---|---|---|---|---|
| `torchvision.models.mobilenetv2.InvertedResidual` | constant-attribute-gated branch unification (if self.<bool_attr>: ... else ...) where the attribute is fixed by __init__ | fragment | `(B,32,16,16)` | Refuted-Proof |
| `torchvision.models.convnext.LayerNorm2d` | F.layer_norm under permute() composition with constructor-bound normalized_shape and Parameter weight/bias attributes | roadmap | `(B,96,28,28)` | Refuted-Proof |
| `torchvision.models.segmentation.deeplabv3.ASPPPooling` | iter(self) -> ordered iteration over child modules of an nn.Sequential parent class | fragment | `(B,2048,32,32)` | Refuted-Proof |
| `torchvision.models.segmentation.lraspp.LRASPPHead` | Dict[str, Tensor] positional-input destructuring (input['low'], input['high']) | fragment | `{'low':(B,40,64,64), 'high':(B,960,8,8)}` | Refuted-Proof |
| `timm.models.davit.ChannelAttention` | Tensor.unbind(dim) returning a fixed-length tuple of (n-1)-dim tensors (catalogue has chunk/split but not unbind) | roadmap | `(B,196,192)` | Refuted-Proof |
| `timm.models.davit.ChannelAttentionV2` | Tensor.unbind(dim) tuple-shape rule and folding of N**-0.5 with N drawn from shape (scalar power on a symbolic int) | roadmap | `(B,196,192)` | Refuted-Proof |
| `timm.layers.patch_embed.PatchEmbed` | timm._assert(predicate, msg) precondition propagation -- treat asserted shape equality as a downstream refinement | fragment | `(B,3,224,224)` | Refuted-Proof |
| `transformers.models.whisper.modeling_whisper.WhisperPositionalEmbedding` | Tensor.__getitem__ with a slice(int_a,int_b) yielding a sub-row tensor of symbolic length b-a | roadmap | `input_ids=(B,L), past_key_values_length=int` | Refuted-Proof |
| `transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding` | super().forward(...) resolution to inherited nn.Embedding.forward (currently catalogue requires an explicit nn.Embedding(self, ...) call site) | fragment | `input_ids=(B,L)` | Refuted-Proof |
| `transformers.models.bart.modeling_bart.BartScaledWordEmbedding` | super().forward(...) resolution to inherited nn.Embedding.forward (same gap as Bart positional embedding) | fragment | `input_ids=(B,L)` | Refuted-Proof |
| `transformers.models.falcon.modeling_falcon.FalconLinear` | input @ self.weight.T  (Parameter.T = .transpose(-1,-2)) -- the inlined-Linear pattern requires a transposed-Parameter matmul rule | roadmap | `(B,L,4544)` | Refuted-Proof |
| `transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding` | torch.cumsum(attention_mask, dim=1) shape rule + super().forward(...) resolution to inherited nn.Embedding.forward | mixed | `attention_mask=(B,L)` | Refuted-Proof |

## Paper claim closed

Round-3 reviewer W5 raised that the 12/78 upper bound was asserted but not exhibited per-block.  Round-4 reviewer Q3 additionally asked which of the named single missing rules are roadmap operator-rule additions vs fragment-extensions outside the well-typed-operator-rule discipline.  This artefact provides the per-block id -> missing rule -> extension class -> witness shape mapping; 5/12 are roadmap additions and 6/12 are fragment-extensions, with 1/12 mixed.
