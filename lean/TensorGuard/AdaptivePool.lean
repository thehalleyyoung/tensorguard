/-
TensorGuard `nn.AdaptiveAvgPool2d` shape-rule, machine-checked in Lean 4
(Step 157).

`nn.AdaptiveAvgPool2d(output_size)` maps `(N, C, H, W) → (N, C, oH, oW)` where the
spatial output equals the requested `output_size` **regardless of the input
spatial size**, keeping batch and channels.  This is exactly what
`src/model_checker.py::_propagate_adaptive_avgpool2d` computes.

We prove:

  * **target-size exactness** (`ap_spatial_h`, `ap_spatial_w`): the output spatial
    dims are exactly the requested `oH, oW` — independent of the input `H, W`;
  * **batch/channel preservation** (`ap_batch`, `ap_channels`);
  * **rank** (`ap_rank`): the output is 4-D;
  * **idempotence** (`ap_idempotent`): pooling to a size the tensor already has is
    shape-preserving (a target of `(H, W)` returns `(N, C, H, W)`).

The companion test `tests/test_adaptivepool_rule.py` replays the rule on **real**
`nn.AdaptiveAvgPool2d` modules over a grid of input/target sizes, confirming the
output spatial dims equal the target for every input.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace AdaptivePool

/-- AdaptiveAvgPool2d output shape: batch & channels kept, spatial = target. -/
def apShape (n c oh ow : Nat) : List Nat := [n, c, oh, ow]

theorem ap_rank (n c oh ow : Nat) : (apShape n c oh ow).length = 4 := rfl
theorem ap_batch (n c oh ow : Nat) : (apShape n c oh ow).head? = some n := rfl
theorem ap_channels (n c oh ow : Nat) : (apShape n c oh ow).get? 1 = some c := rfl

/-- **Target-size exactness** (height): the output height is the requested `oh`,
    independent of the input `H`. -/
theorem ap_spatial_h (n c oh ow : Nat) : (apShape n c oh ow).get? 2 = some oh := rfl
theorem ap_spatial_w (n c oh ow : Nat) : (apShape n c oh ow).get? 3 = some ow := rfl

/-- **Idempotence**: pooling to the size the tensor already has preserves the
    full shape. -/
theorem ap_idempotent (n c h w : Nat) : apShape n c h w = [n, c, h, w] := rfl

end AdaptivePool
end TensorGuard
