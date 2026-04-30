# Per-guard-kind breakdown of recompiles (5 end-to-end blocks)

Round-4 reviewer Q4.  The Track-E 17-module audit classified
all 48 in-contract recompiles as `{SHAPE:0, DTYPE:0, RANK:0,
INT:48}`.  This file reports the same breakdown for the
smaller end-to-end TG-Verified subset (BasicBlock, Bottleneck,
InvertedResidual, Fire, timm.vit.Block).

## Aggregate (n_subjects = 5)

- recompiles total: **8**
- by guard kind: **{'SHAPE': 8}**
- shape/dtype/rank recompiles: **8**
- of which guard variable is an input-shape refinement variable (in catalogue): **8**
- of which guard variable is outside the catalogue (would falsify Theorem 5): **0**

## Per block

| block | recompiles | by guard kind |
|---|---|---|
| tv_resnet_BasicBlock | 2 | {'SHAPE': 2} |
| tv_resnet_Bottleneck | 2 | {'SHAPE': 2} |
| tv_mnv2_InvertedResidual | 1 | {'SHAPE': 1} |
| tv_squeezenet_Fire | 2 | {'SHAPE': 2} |
| timm_vit_Block | 1 | {'SHAPE': 1} |

Per-line capture parsed real guard expressions; classification via the keyword scheme of dynamo_falsification_audit.py.

Run with `python3.11 reproducibility/dynamo_e2e_guard_kinds.py`.
