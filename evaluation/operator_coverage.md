# PyTorch operator surface coverage matrix

Public operator surface of `torch`, `torch.nn` and `torch.nn.functional` cross-referenced against the operators TensorGuard recognises (the universal transfer-function registry, the `nn.Module` layer map, the functional/torch/method dispatch tables, and the denotational transfer functions). Generated against torch `2.9.1`, Python `3.14`.

| Namespace | Public operators | Covered | Coverage |
|-----------|------------------|---------|----------|
| `torch` | 730 | 89 | 0.122 |
| `torch.nn` | 162 | 68 | 0.420 |
| `torch.nn.functional` | 139 | 23 | 0.166 |
| **total** | 1031 | 180 | 0.175 |

Covered operators per namespace:

* `torch`: abs, acos, add, all, amax, amin, any, argsort, asin, atan, bernoulli, bmm, cat, cdist, ceil, celu, clamp, clip, cos, cosh, cross, detach, einsum, eq, equal, erf, erfc, exp, flatten, floor, gather, ge, gt, hardshrink, index_select, isfinite, isinf, isnan, kron, le, log, log10, log2, logsumexp, lt, matmul, max, mean, min, mm, multinomial, mv, nan_to_num, ne, neg, norm, outer, permute, poisson, prelu, prod, qr, relu, reshape, round, rrelu, rsqrt, scatter, selu, sigmoid, sign, sin, sinh, solve, sort, sqrt, squeeze, stack, std, sum, svd, tan, tanh, tensordot, topk, transpose, unique, unsqueeze, var
* `torch.nn`: AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, AlphaDropout, AvgPool1d, AvgPool2d, BatchNorm1d, BatchNorm2d, BatchNorm3d, ConstantPad2d, Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Dropout, Dropout2d, Dropout3d, ELU, Embedding, Flatten, Fold, FractionalMaxPool2d, GELU, GRU, GroupNorm, Hardsigmoid, Hardswish, Identity, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, LPPool2d, LSTM, LayerNorm, LeakyReLU, Linear, LogSoftmax, MaxPool1d, MaxPool2d, MaxPool3d, Mish, ModuleList, MultiheadAttention, PReLU, PixelShuffle, PixelUnshuffle, RNN, ReLU, ReLU6, ReflectionPad2d, ReplicationPad2d, SELU, Sequential, SiLU, Sigmoid, Softmax, SyncBatchNorm, Tanh, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer, Unfold, Upsample, ZeroPad2d
* `torch.nn.functional`: celu, dropout, elu, gelu, hardshrink, hardsigmoid, hardswish, leaky_relu, log_softmax, logsigmoid, mish, prelu, relu, rrelu, selu, sigmoid, silu, softmax, softplus, softshrink, softsign, tanh, tanhshrink
