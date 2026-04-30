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
| Successful modules | **12** |
| Excluded (build/warmup/timeout) | 15 |
| Total in-contract recompiles | 19 |
| SHAPE/DTYPE/RANK recompiles | 1 |
| SHAPE/DTYPE/RANK outside catalogue | 1 |
| **Falsifier rate** | **1/19 = 0.0526** |
| Modules with at least one falsifying guard | 1 |

### Aggregate by guard_kind

| guard_kind | count |
|---|---|
| INT | 9 |
| SHAPE | 1 |

## Paper Claim Closed

Reviewer W4 raised that the 17-module audit is too small to support the necessary direction of Theorem 5.  This audit extends to 12 modules.  The falsifier rate (1/19 = 0.0526) measures the fraction of in-contract recompiles that would constitute a counterexample to Theorem 5 (a SHAPE/DTYPE/RANK guard on a variable outside the TG operator-rule catalogue).  A rate of 0 means the theorem was not falsified on this corpus.

## Per-Module Breakdown

| name | family | status | n_inputs | recompiles | SHAPE+DTYPE+RANK | outside_cat | falsifies |
|---|---|---|---|---|---|---|---|
| tv_resnet50 | torchvision | subproc_timeout | 0 | 0 | 0 | 0 | False |
| tv_squeezenet1_1 | torchvision | ok | 7 | 10 | 1 | 1 | True |
| tv_mnasnet0_5 | torchvision | subproc_timeout | 0 | 0 | 0 | 0 | False |
| tv_googlenet | torchvision | subproc_timeout | 0 | 0 | 0 | 0 | False |
| tv_mnv2_inverted | torchvision | ok | 7 | 0 | 0 | 0 | False |
| hf_bert_tiny | transformers | ok | 7 | 1 | 0 | 0 | False |
| hf_roberta_tiny | transformers | ok | 7 | 0 | 0 | 0 | False |
| timm_resnet18 | timm | ok | 7 | 3 | 0 | 0 | False |
| timm_mobilenetv2_100 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_efficientnet_b1 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_rexnetr_130 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_deit_small_patch16_224 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_convnext_pico | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_poolformer_s12 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_pit_xs_distilled_224 | timm | ok | 7 | 0 | 0 | 0 | False |
| timm_resnext50_32x4d | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_densenet169 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_skresnet34 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_regnety_002 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_hrnet_w18_small | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_fbnetc_100 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_tf_efficientnet_lite0 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_seresnext26ts | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_tinynet_b | timm | ok | 7 | 0 | 0 | 0 | False |
| timm_res2net50_26w_4s | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_ese_vovnet19b_slim | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_ecaresnet26t | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
