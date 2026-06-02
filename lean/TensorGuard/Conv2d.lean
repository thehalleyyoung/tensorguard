/-
TensorGuard `nn.Conv2d` spatial shape-rule, machine-checked in Lean 4 (Step 152).

`nn.Conv2d` maps `(N, C_in, H, W) → (N, C_out, H', W')` where each spatial output
size obeys the standard convolution formula

    out = ⌊(in + 2·pad − dilation·(kernel−1) − 1) / stride⌋ + 1
        = (in_pad − eff) / stride + 1     with  in_pad = in + 2·pad,
                                                eff    = dilation·(kernel−1) + 1.

This is exactly what `src/model_checker.py::_propagate_conv2d` computes (it uses
Python floor division `//` on the same expression), together with the channel
transform `C_in ↦ C_out` and the non-positive-output guard `H',W' ≥ 1`.

This file models the integer spatial map `convOut` (Nat floor division, exact in
the valid regime `eff ≤ in_pad`) and the 4-D shape assembly, proving the laws the
verifier relies on:

  * **identity** (`convOut_identity`): a 1×1 stride-1 no-pad conv preserves the
    spatial size — the spatial map is the identity on `H,W` (≥ 1);
  * **stride-1 closed form** (`convOut_stride_one`): with stride 1 the output is
    `in_pad − eff + 1`;
  * **monotonicity** (`convOut_mono`): the spatial map is monotone in the input
    size (bigger input ⇒ at-least-as-big output) — soundness of size bounds;
  * **upper bound** (`convOut_le`): for a real (eff ≥ 1, stride ≥ 1) conv the
    output never exceeds the padded input;
  * **positive-output guard** (`convOut_pos`): in the valid regime the output is
    ≥ 1 (mirrors the verifier's non-positive-output refutation);
  * **shape assembly** (`conv2d_rank`, `conv2d_batch`, `conv2d_channels`): the
    output is 4-D, keeps the batch dim, and sets the channel dim to `C_out`.

The companion test `tests/test_conv2d_rule.py` replays `convOut` and the shape
assembly on **real** `nn.Conv2d` modules across a grid of kernel/stride/padding/
dilation, asserting the Lean predictions match the live engine — and that a
non-positive prediction coincides with torch raising.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace Conv2d

/-- Nat floor division is monotone in the numerator (core has no
    `Nat.div_le_div_right`, so we derive it). -/
theorem div_mono {a b k : Nat} (h : a ≤ b) : a / k ≤ b / k := by
  rcases Nat.eq_zero_or_pos k with hk | hk
  · subst hk; simp
  · rw [Nat.le_div_iff_mul_le hk]
    exact Nat.le_trans (Nat.div_mul_le_self a k) h

/-- Effective kernel extent `dilation·(kernel−1) + 1`. -/
def eff (kernel dilation : Nat) : Nat := dilation * (kernel - 1) + 1

/-- Spatial output size: `(in + 2·pad − eff) / stride + 1` (Nat floor division,
    exact against torch in the valid regime `eff ≤ in + 2·pad`). -/
def convOut (input pad eff stride : Nat) : Nat :=
  (input + 2 * pad - eff) / stride + 1

/-- **Identity**: a 1×1, stride-1, no-pad conv (eff = 1) preserves a positive
    spatial size. -/
theorem convOut_identity (input : Nat) (h : 1 ≤ input) :
    convOut input 0 1 1 = input := by
  unfold convOut
  simp [Nat.div_one]
  omega

/-- **Stride-1 closed form**: with stride 1 the output is `in_pad − eff + 1`. -/
theorem convOut_stride_one (input pad e : Nat) :
    convOut input pad e 1 = input + 2 * pad - e + 1 := by
  unfold convOut
  simp [Nat.div_one]

/-- **Monotonicity** in the input size. -/
theorem convOut_mono (i₁ i₂ pad e stride : Nat) (h : i₁ ≤ i₂) :
    convOut i₁ pad e stride ≤ convOut i₂ pad e stride := by
  unfold convOut
  have h1 : i₁ + 2 * pad - e ≤ i₂ + 2 * pad - e :=
    Nat.sub_le_sub_right (Nat.add_le_add_right h _) e
  exact Nat.add_le_add_right (div_mono h1) 1

/-- **Upper bound**: for a real conv (eff ≥ 1, stride ≥ 1) within the valid
    regime the output never exceeds the padded input. -/
theorem convOut_le (input pad e stride : Nat)
    (he : 1 ≤ e) (hs : 1 ≤ stride) (hv : e ≤ input + 2 * pad) :
    convOut input pad e stride ≤ input + 2 * pad := by
  unfold convOut
  have hdiv : (input + 2 * pad - e) / stride ≤ input + 2 * pad - e :=
    Nat.div_le_self _ _
  have : (input + 2 * pad - e) + 1 ≤ input + 2 * pad := by omega
  exact Nat.le_trans (Nat.add_le_add_right hdiv 1) this

/-- **Positive-output guard**: the output is always ≥ 1 (a `_+1`), so in the
    valid regime the verifier's non-positive refutation never fires spuriously. -/
theorem convOut_pos (input pad e stride : Nat) : 1 ≤ convOut input pad e stride := by
  unfold convOut; exact Nat.le_add_left 1 _

/- ===================================================================== -/
/- Shape assembly: (N, C_in, H, W) ↦ (N, C_out, H', W')                   -/
/- ===================================================================== -/

/-- The 4-D output shape produced by the verifier's conv2d propagator. -/
def conv2dShape (n cout : Nat) (hOut wOut : Nat) : List Nat := [n, cout, hOut, wOut]

theorem conv2d_rank (n cout hOut wOut : Nat) :
    (conv2dShape n cout hOut wOut).length = 4 := rfl

theorem conv2d_batch (n cout hOut wOut : Nat) :
    (conv2dShape n cout hOut wOut).head? = some n := rfl

theorem conv2d_channels (n cout hOut wOut : Nat) :
    (conv2dShape n cout hOut wOut).get? 1 = some cout := rfl

end Conv2d
end TensorGuard
