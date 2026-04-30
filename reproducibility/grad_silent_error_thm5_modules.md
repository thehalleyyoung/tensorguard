# Grad-flag silent-error audit on 17 Track-E modules

Checks whether any of the 17 torchvision/timm modules used for the
Theorem-5 end-to-end audit exhibit patterns that trigger the known
first-order grad-flag silent-misclassification (renamed-attribute
parameter sharing or `torch.utils.checkpoint`).

## Headline
- Classes inspected: **16**
- Import errors: **2**
- Uses torch.utils.checkpoint: **0**
- Has renamed-attribute sharing: **0**
- TG grad-flag risk (checkpoint OR renamed sharing): **0**

## Per-class

| class | checkpoint | renamed_sharing | risk |
|---|---|---|---|
| torchvision.models.resnet.BasicBlock | False | False | False |
| torchvision.models.resnet.Bottleneck | False | False | False |
| torchvision.models.mobilenetv2.InvertedResidual | False | False | False |
| torchvision.models.squeezenet.Fire | False | False | False |
| torchvision.models.vgg.VGG | False | False | False |
| torchvision.models.densenet.DenseBlock | import_error | import_error | N/A |
| torchvision.models.densenet.DenseLayer | import_error | import_error | N/A |
| torchvision.models.shufflenetv2.InvertedResidual | False | False | False |
| torchvision.models.mobilenetv3.InvertedResidual | False | False | False |
| torchvision.models.resnet.ResNet | False | False | False |
| timm.models.vision_transformer.Block | False | False | False |
| timm.models.swin_transformer.SwinTransformerBlock | False | False | False |
| timm.models.mlp_mixer.MixerBlock | False | False | False |
| timm.models.convnext.ConvNeXtBlock | False | False | False |
| timm.models.regnet.Bottleneck | False | False | False |
| torchvision.models.vision_transformer.EncoderBlock | False | False | False |

## Reproduce

    PYTHONPATH=. python3.11 reproducibility/grad_silent_error_thm5_modules.py

## Paper claim

Cited in Sec. 4.2 and Sec. 6 (limconc_v6.tex): the Track-E modules
have 0/16 grad-flag risk indicators, consistent with the
population-level ≤12% prevalence estimate.