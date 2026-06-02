# torch.fx frontend trace-success rate (real model zoos)

End-to-end trace and lowering of 21 real `torchvision` architectures into TensorGuard's computation graph, generated against torch `2.9.1`, torchvision `0.24.1`.

Trace-or-lower success: **21 of 21** models lowered without crashing (rate 1.000). Across all models, 4860 of 5238 graph steps are operators the frontend reasons about precisely; the remaining 378 are soundly abstracted as unsupported (Step 34).

| Model | Traced | Lowered | Steps | Unsupported |
|-------|--------|---------|-------|-------------|
| `resnet18` | yes | yes | 69 | 0 |
| `resnet50` | yes | yes | 175 | 0 |
| `resnext50_32x4d` | yes | yes | 175 | 0 |
| `wide_resnet50_2` | yes | yes | 175 | 0 |
| `vgg11` | yes | yes | 30 | 0 |
| `vgg16` | yes | yes | 40 | 0 |
| `alexnet` | yes | yes | 22 | 0 |
| `squeezenet1_0` | yes | yes | 66 | 0 |
| `densenet121` | yes | yes | 431 | 0 |
| `mobilenet_v2` | yes | yes | 153 | 0 |
| `mobilenet_v3_small` | yes | yes | 157 | 0 |
| `mnasnet1_0` | yes | yes | 153 | 0 |
| `shufflenet_v2_x1_0` | yes | yes | 368 | 32 |
| `efficientnet_b0` | yes | yes | 249 | 9 |
| `regnet_y_400mf` | yes | yes | 270 | 0 |
| `convnext_tiny` | yes | yes | 202 | 18 |
| `googlenet` | yes | yes | 197 | 0 |
| `inception_v3` | yes | yes | 314 | 0 |
| `vit_b_16` | yes | yes | 232 | 60 |
| `swin_t` | yes | yes | 173 | 51 |
| `maxvit_t` | yes | yes | 1587 | 208 |
