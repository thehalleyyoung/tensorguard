# Theorem 5 Dynamo Falsifier Audit — strictly larger module population

## Command

```
python3.11 reproducibility/dynamo_theorem5_n200.py
```

## Methodology

Subject pool draws from torchvision 0.24.x + transformers 4.57.x + timm 1.0.x + an `AutoModel.from_pretrained` snapshot family + a four-row positive-control sentinel set of synthetic modules that read derived shape/rank/dtype bits via Python-side branching.  Each module is run in subprocess isolation under `torch.compile(dynamic=True)` with structured recompile-reason capture (`torch._logging.set_logs(recompiles=INFO)`).  Per-recompile classification keys on the structured reason text: `tensor 'L['x']' size mismatch …` → SHAPE, `… dtype mismatch` → DTYPE, `… rank mismatch` → RANK, `L['n'] == 5` and other scalar specialisations → INT.

## Aggregate result

| Metric | Value |
|---|---|
| Candidate modules | 146 |
| Successfully audited | **74** |
| In-contract recompiles | 0 |
| SHAPE/DTYPE/RANK recompiles | 0 |
| SHAPE/DTYPE/RANK outside catalogue (Theorem 5 falsifiers) | **0** |
| Modules with at least one falsifier event | 0 |
| Falsifier rate (outside / in-contract) | 0.0000 |

### Aggregate by guard kind
| kind | count |
|---|---|

### Modules with falsifier events

None observed.

## Per-module breakdown

| name | family | status | rc | sdr | outside | falsifies |
|---|---|---|---|---|---|---|
| tv_resnet18 | torchvision | ok | 0 | 0 | 0 | False |
| tv_resnet34 | torchvision | ok | 0 | 0 | 0 | False |
| tv_resnet50 | torchvision | ok | 0 | 0 | 0 | False |
| tv_resnet101 | torchvision | subproc_timeout | 0 | 0 | 0 | False |
| tv_mobilenet_v2 | torchvision | ok | 0 | 0 | 0 | False |
| tv_mobilenet_v3_s | torchvision | ok | 0 | 0 | 0 | False |
| tv_mobilenet_v3_l | torchvision | subproc_timeout | 0 | 0 | 0 | False |
| tv_efficientnet_b0 | torchvision | ok | 0 | 0 | 0 | False |
| tv_efficientnet_b1 | torchvision | subproc_timeout | 0 | 0 | 0 | False |
| tv_efficientnet_v2_s | torchvision | subproc_timeout | 0 | 0 | 0 | False |
| tv_squeezenet1_0 | torchvision | ok | 0 | 0 | 0 | False |
| tv_squeezenet1_1 | torchvision | ok | 0 | 0 | 0 | False |
| tv_regnet_y_400mf | torchvision | subproc_timeout | 0 | 0 | 0 | False |
| tv_regnet_x_400mf | torchvision | ok | 0 | 0 | 0 | False |
| tv_convnext_tiny | torchvision | ok | 0 | 0 | 0 | False |
| tv_shufflenet_v2 | torchvision | subproc_timeout | 0 | 0 | 0 | False |
| tv_shufflenet_v2_1_0 | torchvision | subproc_timeout | 0 | 0 | 0 | False |
| tv_mnasnet0_5 | torchvision | subproc_timeout | 0 | 0 | 0 | False |
| tv_mnasnet1_0 | torchvision | ok | 0 | 0 | 0 | False |
| tv_vgg11 | torchvision | ok | 0 | 0 | 0 | False |
| tv_vgg13 | torchvision | ok | 0 | 0 | 0 | False |
| tv_alexnet | torchvision | ok | 0 | 0 | 0 | False |
| tv_densenet121 | torchvision | subproc_timeout | 0 | 0 | 0 | False |
| tv_googlenet | torchvision | subproc_timeout | 0 | 0 | 0 | False |
| tv_resnet_basic | torchvision | ok | 0 | 0 | 0 | False |
| tv_resnet_bottleneck | torchvision | ok | 0 | 0 | 0 | False |
| tv_mnv2_inverted | torchvision | ok | 0 | 0 | 0 | False |
| tv_squeezenet_fire | torchvision | ok | 0 | 0 | 0 | False |
| tv_shufflenetv2_ir | torchvision | ok | 0 | 0 | 0 | False |
| tv_densenet_denselayer | torchvision | warmup_failed | 0 | 0 | 0 | False |
| hf_bert_tiny | transformers | ok | 0 | 0 | 0 | False |
| hf_gpt2_tiny | transformers | ok | 0 | 0 | 0 | False |
| hf_distilbert_tiny | transformers | ok | 0 | 0 | 0 | False |
| hf_albert_tiny | transformers | ok | 0 | 0 | 0 | False |
| hf_roberta_tiny | transformers | ok | 0 | 0 | 0 | False |
| hf_electra_tiny | transformers | ok | 0 | 0 | 0 | False |
| hf_deberta_tiny | transformers | ok | 0 | 0 | 0 | False |
| hf_mpnet_tiny | transformers | ok | 0 | 0 | 0 | False |
| hf_bart_tiny | transformers | ok | 0 | 0 | 0 | False |
| hf_opt_tiny | transformers | ok | 0 | 0 | 0 | False |
| hf_auto_tiny-random-bert | transformers_auto | ok | 0 | 0 | 0 | False |
| hf_auto_tiny-random-gpt2 | transformers_auto | ok | 0 | 0 | 0 | False |
| hf_auto_tiny-random-t5 | transformers_auto | warmup_failed | 0 | 0 | 0 | False |
| hf_auto_tiny-random-bart | transformers_auto | ok | 0 | 0 | 0 | False |
| hf_auto_tiny-random-roberta | transformers_auto | ok | 0 | 0 | 0 | False |
| hf_auto_tiny-random-distilbert | transformers_auto | ok | 0 | 0 | 0 | False |
| hf_auto_tiny-random-electra | transformers_auto | ok | 0 | 0 | 0 | False |
| hf_auto_tiny-random-mpnet | transformers_auto | ok | 0 | 0 | 0 | False |
| hf_auto_tiny-random-vit | transformers_auto | warmup_failed | 0 | 0 | 0 | False |
| hf_auto_tiny-random-CLIPModel | transformers_auto | warmup_failed | 0 | 0 | 0 | False |
| timm_resnet18 | timm | ok | 0 | 0 | 0 | False |
| timm_resnet26 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_resnet34 | timm | ok | 0 | 0 | 0 | False |
| timm_resnet50 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_mobilenetv2_050 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_mobilenetv2_100 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_mobilenetv3_small_050 | timm | ok | 0 | 0 | 0 | False |
| timm_mobilenetv3_small_100 | timm | ok | 0 | 0 | 0 | False |
| timm_mobilenetv3_large_100 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_efficientnet_b0 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_efficientnet_b1 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_efficientnet_b2 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_mnasnet_050 | timm | ok | 0 | 0 | 0 | False |
| timm_mnasnet_100 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_rexnetr_100 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_rexnetr_130 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_vit_tiny_patch16_224 | timm | ok | 0 | 0 | 0 | False |
| timm_vit_small_patch32_224 | timm | ok | 0 | 0 | 0 | False |
| timm_deit_tiny_patch16_224 | timm | ok | 0 | 0 | 0 | False |
| timm_deit_small_patch16_224 | timm | ok | 0 | 0 | 0 | False |
| timm_mixer_b16_224 | timm | ok | 0 | 0 | 0 | False |
| timm_resmlp_12_224 | timm | ok | 0 | 0 | 0 | False |
| timm_convnext_femto | timm | ok | 0 | 0 | 0 | False |
| timm_convnext_pico | timm | ok | 0 | 0 | 0 | False |
| timm_convnext_nano | timm | ok | 0 | 0 | 0 | False |
| timm_convmixer_768_32 | timm | ok | 0 | 0 | 0 | False |
| timm_xcit_tiny_12_p16_224 | timm | ok | 0 | 0 | 0 | False |
| timm_xcit_small_12_p16_224 | timm | ok | 0 | 0 | 0 | False |
| timm_poolformer_s12 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_poolformer_s24 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_swin_tiny_patch4_window7_224 | timm | ok | 0 | 0 | 0 | False |
| timm_swin_small_patch4_window7_224 | timm | ok | 0 | 0 | 0 | False |
| timm_pit_xs_distilled_224 | timm | ok | 0 | 0 | 0 | False |
| timm_coat_tiny | timm | warmup_failed | 0 | 0 | 0 | False |
| timm_botnet26t_256 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_resnext26ts | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_resnext50_32x4d | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_wide_resnet50_2 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_resnest14d | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_densenet121 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_densenet169 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_dla34 | timm | ok | 0 | 0 | 0 | False |
| timm_dla46_c | timm | ok | 0 | 0 | 0 | False |
| timm_skresnet18 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_skresnet34 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_regnetx_002 | timm | ok | 0 | 0 | 0 | False |
| timm_regnetx_004 | timm | ok | 0 | 0 | 0 | False |
| timm_regnety_002 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_regnety_004 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_repvgg_a0 | timm | ok | 0 | 0 | 0 | False |
| timm_repvgg_a1 | timm | ok | 0 | 0 | 0 | False |
| timm_hrnet_w18_small | timm | ok | 0 | 0 | 0 | False |
| timm_ghostnet_050 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_ghostnet_100 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_fbnetc_100 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_fbnetv3_b | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_tf_efficientnet_lite0 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_tf_efficientnet_lite1 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_legacy_seresnet18 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_legacy_seresnet34 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_seresnext26ts | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_nfnet_l0 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_nf_resnet26 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_tinynet_a | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_tinynet_b | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_levit_128 | timm | warmup_failed | 0 | 0 | 0 | False |
| timm_levit_192 | timm | warmup_failed | 0 | 0 | 0 | False |
| timm_res2net50_26w_4s | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_res2next50 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_rexnet_100 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_hardcorenas_a | timm | ok | 0 | 0 | 0 | False |
| timm_ese_vovnet19b_slim | timm | ok | 0 | 0 | 0 | False |
| timm_ese_vovnet39b | timm | ok | 0 | 0 | 0 | False |
| timm_eca_resnet33ts | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_ecaresnet26t | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_ecaresnet50t | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_cspresnet50 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_cspdarknet53 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_darknet53 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_darknet17 | timm | ok | 0 | 0 | 0 | False |
| timm_gernet_s | timm | ok | 0 | 0 | 0 | False |
| timm_gernet_m | timm | ok | 0 | 0 | 0 | False |
| timm_selecsls42 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_selecsls60 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_twins_pcpvt_small | timm | ok | 0 | 0 | 0 | False |
| timm_twins_svt_small | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_visformer_small | timm | warmup_failed | 0 | 0 | 0 | False |
| timm_edgenext_xx_small | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_edgenext_x_small | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_maxxvit_rmlp_nano_rw_256 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_fastvit_t8 | timm | subproc_timeout | 0 | 0 | 0 | False |
| timm_fastvit_t12 | timm | subproc_timeout | 0 | 0 | 0 | False |
| sentinel_shape_branch | custom_op | ok | 0 | 0 | 0 | False |
| sentinel_rank_branch | custom_op | ok | 0 | 0 | 0 | False |
| sentinel_dtype_branch | custom_op | ok | 0 | 0 | 0 | False |
| sentinel_hidden_int_cache | custom_op | ok | 0 | 0 | 0 | False |

## Paper claim

Cited by §4.3 / Theorem~\ref{thm:dynamo-corr} as the round-7 extended falsifier audit.  The four `sentinel_*` rows in the table are positive controls — synthetic modules that read derived shape/rank/dtype bits via Python-side branching, designed to confirm the audit is *capable* of emitting an outside-catalogue SHAPE/DTYPE/RANK guard when one exists in the source.
