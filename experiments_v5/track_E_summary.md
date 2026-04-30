# Track E v5 — TorchDynamo guard correspondence

**Script:** `experiments_v5/run_dynamo_correspondence_v5.py`
**Raw results:** `experiments_v5/dynamo_correspondence_v5.json`
**Paper section:** `docs/paper/sections_v5/E_dynamo.tex`

## Setup
- CPU-only, `python3.11`
- Pinned: torch 2.9.1, torchvision 0.24.1, transformers 4.57.3, timm 1.0.26
- 17 modules: 8 torchvision (resnet18/50, mobilenet_v3_small,
  efficientnet_b0, squeezenet1_1, regnet_y_400mf, vit_b_16, convnext_tiny),
  5 transformers (BERT/GPT2/T5/DistilBERT/ViT, all instantiated from
  *tiny* configs to fit memory, random init), 3 timm (deit_tiny_p16_224,
  mobilenetv3_small_050, resnet18), and 1 TG-verified surrogate
  (`TinyMLP`) for cross-checking the SAFE verdict.
- For each module: `torch.compile(M, dynamic=True)`, 3 warm-up draws to
  fix the dynamic graph, then 32 in-contract draws (varying free
  symbolic dims B, S, H, W within ranges) and 3 out-of-contract
  positive-control draws (changed channel, dtype, rank).
- Recompiles are read from
  `torch._dynamo.utils.counters["stats"]["unique_graphs"]`.

## Headline numbers
| metric | value |
|---|---|
| modules built and run | 17/17 |
| modules with **0** in-contract recompiles | **6/17** |
| in-contract recompile rate (per call) | **8.8 %** (48/544) |
| out-of-contract violation detection rate | **97.9 %** (46/47) |

## Calibrated finding
TensorGuard's shape contract is **necessary but not sufficient** for
TorchDynamo guard-stability:

- **Sufficient on tensor-pure forwards.** `efficientnet_b0`, `vit_b_16`,
  `convnext_tiny`, `hf_vit_tiny`, `deit_tiny_p16_224` and
  `mobilenetv3_small_050` show 0 in-contract recompiles across 32 calls
  each — the predicted behaviour.
- **Insufficient when Python-level integer arithmetic depends on a
  symbolic dimension.** `squeezenet1_1` (26/32), `tv_resnet18` (5/32),
  `tv_resnet50` (4/32), `timm_resnet18` (4/32) and the HF text models
  (1–3/32) trigger extra Dynamo specialisations from
  `(in_size + 2*pad − k)//stride + 1` arithmetic and `.view(B, C, -1)`
  reshapes that Dynamo's shape-environment refuses to unify.
- **Positive control fires almost always.** 46/47 contract violations
  are caught either by a Dynamo recompile (HF text models) or a runtime
  shape/dtype check (vision models with eager fallback). The single
  miss is rank-promotion (2-D → 3-D) on a `Linear`, which broadcasts
  cleanly — a Linear semantics quirk, not a Dynamo failure.

## What this means for the paper
The empirical contrapositive of the sufficient-condition statement holds
for the **static-shape fragment** TensorGuard certifies. A fully
sufficient condition would additionally require shape-polymorphism of
the Python control flow surrounding tensor ops — out of scope here, but
motivates the tighter `symbolic_shapes` integration listed in Future
Work.

All raw guard messages, seeds, and per-module breakdown live in the JSON.
