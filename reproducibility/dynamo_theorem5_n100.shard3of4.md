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
| Successful modules | **16** |
| Excluded (build/warmup/timeout) | 10 |
| Total in-contract recompiles | 19 |
| SHAPE/DTYPE/RANK recompiles | 0 |
| SHAPE/DTYPE/RANK outside catalogue | 0 |
| **Falsifier rate** | **0/19 = 0.0000** |
| Modules with at least one falsifying guard | 0 |

### Aggregate by guard_kind

| guard_kind | count |
|---|---|
| INT | 19 |

## Paper Claim Closed

Reviewer W4 raised that the 17-module audit is too small to support the necessary direction of Theorem 5.  This audit extends to 16 modules.  The falsifier rate (0/19 = 0.0000) measures the fraction of in-contract recompiles that would constitute a counterexample to Theorem 5 (a SHAPE/DTYPE/RANK guard on a variable outside the TG operator-rule catalogue).  A rate of 0 means the theorem was not falsified on this corpus.

## Per-Module Breakdown

| name | family | status | n_inputs | recompiles | SHAPE+DTYPE+RANK | outside_cat | falsifies |
|---|---|---|---|---|---|---|---|
| tv_mobilenet_v3_s | torchvision | ok | 7 | 2 | 0 | 0 | False |
| tv_convnext_tiny | torchvision | warmup_failed | 0 | 0 | 0 | 0 | False |
| tv_alexnet | torchvision | warmup_failed | 0 | 0 | 0 | 0 | False |
| tv_resnet_basic | torchvision | ok | 7 | 0 | 0 | 0 | False |
| tv_shufflenetv2_ir | torchvision | ok | 7 | 0 | 0 | 0 | False |
| hf_distilbert_tiny | transformers | ok | 7 | 0 | 0 | 0 | False |
| hf_deberta_tiny | transformers | ok | 7 | 1 | 0 | 0 | False |
| timm_resnet34 | timm | ok | 7 | 3 | 0 | 0 | False |
| timm_mobilenetv3_small_100 | timm | ok | 7 | 3 | 0 | 0 | False |
| timm_mnasnet_100 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_vit_small_patch32_224 | timm | ok | 7 | 0 | 0 | 0 | False |
| timm_resmlp_12_224 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_xcit_small_12_p16_224 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_swin_tiny_patch4_window7_224 | timm | ok | 7 | 0 | 0 | 0 | False |
| timm_botnet26t_256 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_resnest14d | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_dla46_c | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_regnetx_002 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_repvgg_a0 | timm | ok | 7 | 2 | 0 | 0 | False |
| timm_ghostnet_050 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_lambda_resnet26t | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_legacy_seresnet18 | timm | ok | 7 | 3 | 0 | 0 | False |
| timm_nf_resnet26 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_levit_192 | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_rexnet_100 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_dm_nfnet_f0 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
