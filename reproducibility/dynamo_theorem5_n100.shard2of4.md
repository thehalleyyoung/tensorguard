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
| Successful modules | **10** |
| Excluded (build/warmup/timeout) | 17 |
| Total in-contract recompiles | 12 |
| SHAPE/DTYPE/RANK recompiles | 0 |
| SHAPE/DTYPE/RANK outside catalogue | 0 |
| **Falsifier rate** | **0/12 = 0.0000** |
| Modules with at least one falsifying guard | 0 |

### Aggregate by guard_kind

| guard_kind | count |
|---|---|
| INT | 12 |

## Paper Claim Closed

Reviewer W4 raised that the 17-module audit is too small to support the necessary direction of Theorem 5.  This audit extends to 10 modules.  The falsifier rate (0/12 = 0.0000) measures the fraction of in-contract recompiles that would constitute a counterexample to Theorem 5 (a SHAPE/DTYPE/RANK guard on a variable outside the TG operator-rule catalogue).  A rate of 0 means the theorem was not falsified on this corpus.

## Per-Module Breakdown

| name | family | status | n_inputs | recompiles | SHAPE+DTYPE+RANK | outside_cat | falsifies |
|---|---|---|---|---|---|---|---|
| tv_mobilenet_v2 | torchvision | subproc_timeout | 0 | 0 | 0 | 0 | False |
| tv_regnet_y_400mf | torchvision | subproc_timeout | 0 | 0 | 0 | 0 | False |
| tv_vgg11 | torchvision | warmup_failed | 0 | 0 | 0 | 0 | False |
| tv_inception_v3 | torchvision | warmup_failed | 0 | 0 | 0 | 0 | False |
| tv_squeezenet_fire | torchvision | ok | 7 | 1 | 0 | 0 | False |
| hf_gpt2_tiny | transformers | ok | 7 | 0 | 0 | 0 | False |
| hf_electra_tiny | transformers | ok | 7 | 0 | 0 | 0 | False |
| timm_resnet26 | timm | ok | 7 | 4 | 0 | 0 | False |
| timm_mobilenetv3_small_050 | timm | ok | 7 | 1 | 0 | 0 | False |
| timm_mnasnet_050 | timm | ok | 7 | 2 | 0 | 0 | False |
| timm_vit_tiny_patch16_224 | timm | ok | 7 | 0 | 0 | 0 | False |
| timm_mixer_b16_224 | timm | ok | 7 | 0 | 0 | 0 | False |
| timm_convmixer_768_32 | timm | ok | 7 | 0 | 0 | 0 | False |
| timm_poolformer_s24 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_coat_tiny | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_wide_resnet50_2 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_dla34 | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_gluon_resnet18_v1b | timm | ok | 7 | 4 | 0 | 0 | False |
| timm_regnety_004 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_hrnet_w18 | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_fbnetv3_b | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_tf_efficientnet_lite1 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_nfnet_l0 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_levit_128 | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_res2next50 | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
| timm_ese_vovnet39b | timm | warmup_failed | 0 | 0 | 0 | 0 | False |
| timm_ecaresnet50t | timm | subproc_timeout | 0 | 0 | 0 | 0 | False |
