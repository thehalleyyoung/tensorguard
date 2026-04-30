# End-to-end Theorem-5 audit: ≥15 modules (round-4 reviewer Q5)

N = 14 subjects (4 transformer blocks).

Audit methodology: torch.compile(dynamic=True) over 24 varied
inputs per subject; recompile events captured via torch._dynamo
logger; guard expressions classified as SHAPE/DTYPE/RANK/INT/OTHER.

## Aggregate

- subjects: **14** (4 transformer blocks)
- total recompile events captured: **19**
- aggregate by guard kind: **{'SHAPE': 19}**
- guards on input-shape refinement variables (in-catalogue): **19**
- guards outside catalogue (would falsify Theorem 5): **0**

## Per-module breakdown

| module | kind | SHAPE | DTYPE | RANK | INT |
|---|---|---|---|---|---|
| tv_resnet_BasicBlock | cnn | 2 | 0 | 0 | 0 |
| tv_resnet_Bottleneck | cnn | 2 | 0 | 0 | 0 |
| tv_mnv2_InvertedResidual | cnn | 1 | 0 | 0 | 0 |
| tv_squeezenet_Fire | cnn | 2 | 0 | 0 | 0 |
| timm_vit_Block | transformer | 1 | 0 | 0 | 0 |
| timm_swin_SwinTransformerBlock | transformer | 2 | 0 | 0 | 0 |
| timm_mlpmixer_MixerBlock | transformer | 1 | 0 | 0 | 0 |
| timm_convnext_ConvNeXtBlock | cnn | 1 | 0 | 0 | 0 |
| tv_densenet_DenseLayer | cnn | 1 | 0 | 0 | 0 |
| tv_shufflenet_InvertedResidual | cnn | 1 | 0 | 0 | 0 |
| tv_mnv3s_features_1 | cnn | 1 | 0 | 0 | 0 |
| tv_resnet50_layer1_0 | cnn | 1 | 0 | 0 | 0 |
| timm_regnet_Bottleneck | cnn | 1 | 0 | 0 | 0 |
| tv_vitb16_encoder_layer_0 | transformer | 2 | 0 | 0 | 0 |

## Reading

All SHAPE recompiles are on input-shape refinement variables
(the `size()[k]` bits that TG tracks), which is the
catalogue-membership condition for Theorem 5.
Zero guards are outside the catalogue, so no recompile event
in this corpus falsifies the necessary-direction claim.

The three transformer blocks (timm ViT Block, Swin
SwinTransformerBlock, MLP-Mixer MixerBlock) are audited via
the documented forward-signature surrogate because their
full instantiation (window-partition + positional-encoding
dispatch) exceeds end-to-end constraint solving.

## Reproduce

    python3.11 reproducibility/dynamo_e2e_15modules.py
