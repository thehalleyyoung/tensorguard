# TensorGuard Verifiable Fragment (V_TG)

> **Generated file — do not edit by hand.** Regenerate with `python -m src.verifiable_fragment > VERIFIABLE_FRAGMENT.md`. Single source of truth: `src/verifiable_fragment.py`.

This document formally characterizes the subset of PyTorch `nn.Module` code that TensorGuard can analyze. A module inside the fragment is amenable to sound shape/device/gradient verification; a module outside it is **reported as `UNKNOWN`, never silently passed** (see the fallback policy below).

## Grammar

```
Module      ::=  class C(nn.Module):
                     __init__(self, <params>): <InitBody>
                     forward(self, <inputs>): <FwdBody>

InitBody    ::=  (self.<attr> = <LayerExpr>)*
LayerExpr   ::=  nn.<SupportedLayer>(<literal>*)
              |  nn.Sequential(<LayerExpr>*)
              |  nn.ModuleList([<LayerExpr>*])

FwdBody     ::=  <Stmt>*; return <Expr>

Stmt        ::=  <var> = <Expr>
              |  <var> = self.<attr>(<Expr>*)          # submodule call
              |  for <var> in self.<modulelist>:       # static iteration
                     <Stmt>*

Expr        ::=  <var>
              |  self.<attr>(<Expr>*)                  # nn.Module call
              |  <SupportedFunc>(<Expr>*)              # torch.* / F.*
              |  <Expr>.<SupportedMethod>(<literal>*)  # tensor method
              |  <Expr> <BinOp> <Expr>                 # +, *, @
              |  <literal>
```

## Supported constructs

### Layer types (70)

```
AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d,
AlphaDropout, AvgPool1d, AvgPool2d, BatchNorm1d, BatchNorm2d, BatchNorm3d,
ConstantPad2d, Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d,
ConvTranspose3d, Dropout, Dropout2d, Dropout3d, ELU, Embedding, Flatten,
Fold, FractionalMaxPool2d, GELU, GRU, GroupNorm, Hardsigmoid, Hardswish,
Identity, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, LPPool2d, LSTM,
LayerNorm, LeakyReLU, Linear, LogSoftmax, MaxPool1d, MaxPool2d, MaxPool3d,
Mish, ModuleDict, ModuleList, MultiheadAttention, PReLU, PixelShuffle,
PixelUnshuffle, RNN, ReLU, ReLU6, ReflectionPad2d, ReplicationPad2d, SELU,
Sequential, SiLU, Sigmoid, Softmax, SyncBatchNorm, Tanh,
TransformerDecoder, TransformerDecoderLayer, TransformerEncoder,
TransformerEncoderLayer, Unflatten, Unfold, Upsample, ZeroPad2d
```

### Tensor methods (39)

```
add, add_, bmm, chunk, clone, contiguous, cpu, cuda, detach, dim, double,
expand, flatten, float, half, matmul, mean, mm, mul, mul_, narrow, numel,
permute, relu, repeat, reshape, select, shape, sigmoid, size, softmax,
split, squeeze, sum, tanh, to, transpose, unsqueeze, view
```

### torch.* functions (28)

```
torch.add, torch.arange, torch.bmm, torch.cat, torch.chunk,
torch.column_stack, torch.dropout, torch.dstack, torch.einsum,
torch.flatten, torch.hstack, torch.linspace, torch.matmul, torch.mm,
torch.mul, torch.ones, torch.ones_like, torch.relu, torch.row_stack,
torch.sigmoid, torch.softmax, torch.split, torch.stack, torch.tanh,
torch.vstack, torch.where, torch.zeros, torch.zeros_like
```

### torch.nn.functional (F.*) functions (21)

```
F.adaptive_avg_pool2d, F.avg_pool2d, F.batch_norm, F.conv1d, F.conv2d,
F.dropout, F.elu, F.gelu, F.group_norm, F.interpolate, F.layer_norm,
F.leaky_relu, F.linear, F.log_softmax, F.max_pool2d, F.pad, F.relu,
F.sigmoid, F.silu, F.softmax, F.tanh
```

## Excluded constructs (outside V_TG)

| Category | Description | Detected by |
| --- | --- | --- |
| `DATA_DEPENDENT_CONTROL_FLOW` | Branch (if/while) whose condition depends on a tensor value. | static+fx |
| `DATA_DEPENDENT_ITERATION` | Loop whose trip count depends on runtime data (e.g. range(int(x.item()))). | static+fx |
| `DYNAMIC_ASSERTION` | assert statement in forward (may reference tensor values). | static+fx |
| `TENSOR_TO_SCALAR` | .item() / .tolist() / .numpy() converts a tensor to a Python value. | static+fx |
| `CUSTOM_AUTOGRAD` | Custom torch.autograd.Function subclass with opaque shape behaviour. | fx |
| `INPLACE_MUTATION` | In-place mutation that torch.fx cannot trace soundly. | fx |
| `JIT_SCRIPT` | torch.jit.script / scripted submodule opaque to torch.fx. | fx |
| `OPAQUE_EXTERNAL_CALL` | Call into an external/undefined symbol opaque to the tracer. | fx |
| `DYNAMIC_MODULE_CONSTRUCTION` | Submodules constructed dynamically at forward time. | fx |
| `UNSUPPORTED_BUILTIN` | Python builtin not modelled by the shape semantics. | fx |
| `OTHER` | Any other torch.fx trace failure not otherwise classified. | fx |

*Detected by:* `static` = instance-free AST scan (`analyze_source`); `fx` = `torch.fx` trace-error classification during `check_traceability`; `static+fx` = both.

## Fallback policy: unsupported → `UNKNOWN`, never a silent pass

When a module contains any construct above, TensorGuard does **not** emit a confident `SAFE`. Two complementary mechanisms enforce this:

1. **Pre-verification** — `check_traceability(module)` returns `in_verifiable_fragment=False` with the offending `UnsupportedConstruct`s. The instance-free `analyze_source(source)` exposes the statically detectable subset (`DATA_DEPENDENT_CONTROL_FLOW`, `DATA_DEPENDENT_ITERATION`, `DYNAMIC_ASSERTION`, `TENSOR_TO_SCALAR`).
2. **During verification** — in `--soundness-mode sound`, `verify_architecture` folds these signals (plus opaque out-of-fragment layers and heuristic-tagged operators) into abstention, yielding `verdict=UNKNOWN` with `unknown_reasons` rather than a silent `SAFE`. See `SOUNDNESS_CONTRACT.md`.

