# 488-block analysis: unbind handler

## Headline triple
- Verified: 55
- Refuted-Proof (RP): 15
- Abstain: 418
- Total: 488

## Refuted blocks
- `InvertedResidual` (torchvision): [DEAD-OUTPUT] Result 'x1' is computed but never used or returned
- `MNASNet` (torchvision): Potential None subscript on 'depths' without guard
- `WindowPartition` (torchvision): Potential division by zero: 'P' not guarded
- `LRASPPHead` (torchvision): [DEVICE-MISMATCH] Z3 violation (device_mismatch) at step 5:
  phase_s4 = TRAIN_87
  phase_s3 = TRAIN
- `ConvNeXtStage` (timm): Index 0 on 'dilation' without length guard (may be empty)
- `RelativePositionBias` (timm): Index 0 on 'window_size' without length guard (may be empty)
- `TalkingHeadAttn` (timm): Potential division by zero: 'num_heads' not guarded
- `PositionalEncodingFourier` (timm): Potential division by zero: 'dim_t' not guarded
- `ChannelAttention` (timm): [SHAPE-INCOMPATIBLE] Reshape incompatible: cannot reshape TensorShape(dims=(1, 32, 64, 32)) to (1, 6
- `ChannelAttentionV2` (timm): [SHAPE-INCOMPATIBLE] Reshape incompatible: cannot reshape TensorShape(dims=(1, 32, 64, 32)) to (1, 6
- `PatchEmbed` (timm): [SHAPE-INCOMPATIBLE] Conv2d expects 4D input, got 3D
- `CrossAttention` (timm): Potential division by zero: 'num_heads' not guarded
- `BartLearnedPositionalEmbedding` (transformers): [DEAD-OUTPUT] Result 'bsz' is computed but never used or returned
- `Transformer` (transformers): [DEAD-OUTPUT] Result 'attentions' is computed but never used or returned
- `AlbertLayerGroup` (transformers): [USE-BEFORE-DEF] Variable 'outputs' used before definition at line 33

## Change from prior run
Prior headline: 57V / 0RP / 431A (0 unconditional RP).
New headline: 55V / 15RP / 418A.
