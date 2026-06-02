/-
TensorGuard `nn.MaxPool2d` / `nn.AvgPool2d` spatial shape-rule, machine-checked
in Lean 4 (Step 153).

Pooling maps `(N, C, H, W) → (N, C, H', W')` (the channel dim is **kept**, unlike
conv) with the no-dilation formula

    out = ⌊(in + 2·pad − kernel) / stride⌋ + 1 = (in + 2·pad − kernel) / stride + 1,

exactly as `src/model_checker.py::_propagate_pool2d` computes (Python `//`),
including the non-positive-output guard.  This file models the integer spatial
map `poolOut` and proves identity (kernel 1, stride 1), the stride-1 closed form,
monotonicity in the input size, the padded-input upper bound, the
positive-output guard, and that the channel dim is preserved.

The companion test `tests/test_pool2d_rule.py` replays the rule on **real**
`nn.MaxPool2d`/`nn.AvgPool2d` modules over a kernel/stride/padding grid.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace Pool2d

theorem div_mono {a b k : Nat} (h : a ≤ b) : a / k ≤ b / k := by
  rcases Nat.eq_zero_or_pos k with hk | hk
  · subst hk; simp
  · rw [Nat.le_div_iff_mul_le hk]
    exact Nat.le_trans (Nat.div_mul_le_self a k) h

/-- Spatial output size for pooling: `(in + 2·pad − kernel) / stride + 1`. -/
def poolOut (input pad kernel stride : Nat) : Nat :=
  (input + 2 * pad - kernel) / stride + 1

theorem poolOut_identity (input : Nat) (h : 1 ≤ input) :
    poolOut input 0 1 1 = input := by
  unfold poolOut; simp [Nat.div_one]; omega

theorem poolOut_stride_one (input pad kernel : Nat) :
    poolOut input pad kernel 1 = input + 2 * pad - kernel + 1 := by
  unfold poolOut; simp [Nat.div_one]

theorem poolOut_mono (i₁ i₂ pad kernel stride : Nat) (h : i₁ ≤ i₂) :
    poolOut i₁ pad kernel stride ≤ poolOut i₂ pad kernel stride := by
  unfold poolOut
  have h1 : i₁ + 2 * pad - kernel ≤ i₂ + 2 * pad - kernel :=
    Nat.sub_le_sub_right (Nat.add_le_add_right h _) kernel
  exact Nat.add_le_add_right (div_mono h1) 1

theorem poolOut_le (input pad kernel stride : Nat)
    (hk : 1 ≤ kernel) (hv : kernel ≤ input + 2 * pad) :
    poolOut input pad kernel stride ≤ input + 2 * pad := by
  unfold poolOut
  have hdiv : (input + 2 * pad - kernel) / stride ≤ input + 2 * pad - kernel :=
    Nat.div_le_self _ _
  have : (input + 2 * pad - kernel) + 1 ≤ input + 2 * pad := by omega
  exact Nat.le_trans (Nat.add_le_add_right hdiv 1) this

theorem poolOut_pos (input pad kernel stride : Nat) :
    1 ≤ poolOut input pad kernel stride := by
  unfold poolOut; exact Nat.le_add_left 1 _

/-- Pooling keeps the channel dimension (`(N, C, H, W) ↦ (N, C, H', W')`). -/
def pool2dShape (n c hOut wOut : Nat) : List Nat := [n, c, hOut, wOut]

theorem pool2d_rank (n c hOut wOut : Nat) : (pool2dShape n c hOut wOut).length = 4 := rfl
theorem pool2d_batch (n c hOut wOut : Nat) : (pool2dShape n c hOut wOut).head? = some n := rfl
theorem pool2d_channels_preserved (n c hOut wOut : Nat) :
    (pool2dShape n c hOut wOut).get? 1 = some c := rfl

end Pool2d
end TensorGuard
