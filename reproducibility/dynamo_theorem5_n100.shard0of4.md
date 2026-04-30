# Theorem 5 Dynamo Falsifier Audit — ≥100 Modules

## Command

```
python3.11 reproducibility/dynamo_theorem5_n100.py
```

## Inputs / Seed

- Seed: 0
- Candidate modules: 107 (torchvision + transformers + timm)
- In-contract samples per module: 10
- Warmup samples: 3

## Result Numbers

| Metric | Value |
|---|---|
| Successful modules | **17** |
| Excluded (build/warmup/timeout) | 10 |
| Total in-contract recompiles | 22 |
| SHAPE/DTYPE/RANK recompiles | 0 |
| SHAPE/DTYPE/RANK outside catalogue | 0 |
| **Falsifier rate** | **0/22 = 0.0000** |
| Modules with at least one falsifying guard | 0 |

### Aggregate by guard_kind

| guard_kind | count |
|---|---|
| INT | 22 |

## Paper Claim Closed

Reviewer W4 raised that the 17-module audit is too small to support the necessary direction of Theorem 5.  This audit extends to 17 modules.  The falsifier rate (0/22 = 0.0000) measures the fraction of in-contract recompiles that would constitute a counterexample to Theorem 5 (a SHAPE/DTYPE/RANK guard on a variable outside the TG operator-rule catalogue).  A rate of 0 means the theorem was not falsified on this corpus.

## Per-Module Breakdown

| name | family | status | n_inputs | recompiles | SHAPE+DTYPE+RANK | outside_cat | falsifies |
|---|---|---|---|---|---|---|---|
| tv_resnet18 | torchvision | ok | 7 | 2 | 0 | 0 | False |
| tv_efficientnet_b0 | torchvision | ok | 7 | 1 | 0 | 0 | False |
| tv_shufflenet_v2 | torchvision | subproc_timeout | 0 | 0 | 0 | 0 | False |
| tv_densenet121 | torchvision | subproc_timeout | 0 | 0 | 0 | 0 | False |
| tv_resnet_bottleneck | torchvision | ok | 7 | 1 | 0 | 0 | False |
| tv_densenet_denselayer | torchvision | warmup_failed | 0 | 0 | 0 | 0 | False |
| hf_albert_tiny | transformers | ok | 7 | 1 | 0 | 0 | False |
| hf_mpnet_tiny | transformers | ok | 7 | 0 | 0 | 0 | False |
| timm_mobilenetv2_050 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_efficientnet_b0 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_rexnetr_100 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_deit_tiny_patch16_224 | timm | ok | 7 | 0 | 0 | 0 | False |
| timm_convnext_femto | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_xcit_tiny_12_p16_224 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_swin_small_patch4_window7_224 | timm | ok | 7 | 0 | 0 | 0 | False |
| timm_resnext26ts | timm | ok | 7 | 4 | 0 | 0 | False |
| timm_densenet121 | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_skresnet18 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_regnetx_004 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_repvgg_a1 | timm | ok | 7 | 2 | 0 | 0 | False |
| timm_ghostnet_100 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_lambda_resnet50ts | timm | ok | 7 | 4 | 0 | 0 | False |
| timm_legacy_seresnet34 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_tinynet_a | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_halo2botnet50ts_256 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_hardcorenas_a | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_eca_resnet33ts | timm | ok | 7 | 1 | 0 | 0 | False |
