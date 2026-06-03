# Real-model deployment gallery

Each gallery model is a real `nn.Module` that runs under eager PyTorch, then passes a pre-export FX gate and a post-export `torch.export` gate when that backend is available.

## Model gallery

| Model | Family | Inputs | Output | Operator surface |
|-------|--------|--------|--------|------------------|
| `resnet_residual_stage` | ResNet | `x:(1, 3, 32, 32)` | `(1, 5)` | `AdaptiveAvgPool2d`, `BatchNorm2d`, `Conv2d`, `Linear`, `ResidualAdd`, `flatten`, `relu` |
| `vit_patch_mixer` | ViT | `x:(1, 3, 32, 32)` | `(1, 3)` | `Conv2d`, `LayerNorm`, `Linear`, `flatten`, `gelu`, `mean`, `transpose` |
| `llama_style_mlp_block` | Llama-style block | `tokens:(2, 8)` | `(2, 16)` | `Embedding`, `LayerNorm`, `Linear`, `mean`, `mul`, `silu` |
| `diffusion_unet_skip` | Diffusion U-Net | `x:(1, 4, 16, 16)` | `(1, 4, 16, 16)` | `Conv2d`, `ConvTranspose2d`, `cat`, `relu` |
| `recommender_two_tower` | Recommender | `dense:(4, 3)`, `item_ids:(4, 3)`, `user_ids:(4, 3)` | `(4, 1)` | `Embedding`, `Linear`, `cat`, `mean`, `relu` |
| `speech_conv_gru_encoder` | Speech | `x:(2, 80, 32)` | `(2, 12)` | `Conv1d`, `GRU`, `Linear`, `mean`, `relu`, `transpose` |

## Export gate matrix

| Model | Phase | Backend | Required | Gate |
|-------|-------|---------|----------|------|
| `diffusion_unet_skip` | after_export | `torch.export` | if available | `src.export_extractor.verify_module_export` |
| `diffusion_unet_skip` | before_export | `fx` | yes | `src.fx_extractor.verify_module(backend='fx')` |
| `llama_style_mlp_block` | after_export | `torch.export` | if available | `src.export_extractor.verify_module_export` |
| `llama_style_mlp_block` | before_export | `fx` | yes | `src.fx_extractor.verify_module(backend='fx')` |
| `recommender_two_tower` | after_export | `torch.export` | if available | `src.export_extractor.verify_module_export` |
| `recommender_two_tower` | before_export | `fx` | yes | `src.fx_extractor.verify_module(backend='fx')` |
| `resnet_residual_stage` | after_export | `torch.export` | if available | `src.export_extractor.verify_module_export` |
| `resnet_residual_stage` | before_export | `fx` | yes | `src.fx_extractor.verify_module(backend='fx')` |
| `speech_conv_gru_encoder` | after_export | `torch.export` | if available | `src.export_extractor.verify_module_export` |
| `speech_conv_gru_encoder` | before_export | `fx` | yes | `src.fx_extractor.verify_module(backend='fx')` |
| `vit_patch_mixer` | after_export | `torch.export` | if available | `src.export_extractor.verify_module_export` |
| `vit_patch_mixer` | before_export | `fx` | yes | `src.fx_extractor.verify_module(backend='fx')` |

## Rebuild

```bash
make deployment-gallery
make deployment-gallery-gate
```
