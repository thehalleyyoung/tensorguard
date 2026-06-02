/-
TensorGuard `nn.ConvTranspose1d` length shape-rule, machine-checked in Lean 4
(Step 154).

Transposed convolution (a.k.a. fractionally-strided / "deconv") maps
`(N, C_in, L) → (N, C_out, L')` with

    L' = (L − 1)·stride − 2·pad + dilation·(kernel − 1) + output_padding + 1,

exactly as `src/model_checker.py::_propagate_convtranspose1d` computes.  Unlike
forward conv this map **upsamples** (no floor division).  We model it over `Nat`
(with `dk = dilation·(kernel−1)`) and prove the laws the verifier relies on:

  * **identity** (`ctOut_identity`): a stride-1, no-pad, 1×1 transpose conv
    preserves a positive length;
  * **no-pad closed form** (`ctOut_no_pad`);
  * **monotonicity** (`ctOut_mono`): the length map is monotone in the input
    length;
  * **upsampling lower bound** (`ctOut_ge`): with stride ≥ 1 and no padding the
    output is at least as long as the input (transpose conv never shrinks) —
    the defining "upsample" property;
  * **shape laws** (`ct_rank`, `ct_batch`, `ct_channels`).

The companion test `tests/test_convtranspose_rule.py` replays the rule on
**real** `nn.ConvTranspose1d` modules over a stride/padding/output_padding/
dilation grid.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace ConvTranspose

/-- Transposed-conv output length with `dk = dilation·(kernel−1)`. -/
def ctOut (input stride pad dk op : Nat) : Nat :=
  (input - 1) * stride + dk + op + 1 - 2 * pad

theorem ctOut_identity (input : Nat) (h : 1 ≤ input) :
    ctOut input 1 0 0 0 = input := by
  unfold ctOut; simp; omega

theorem ctOut_no_pad (input stride dk op : Nat) :
    ctOut input stride 0 dk op = (input - 1) * stride + dk + op + 1 := by
  unfold ctOut; simp

theorem ctOut_mono (i₁ i₂ stride pad dk op : Nat) (h : i₁ ≤ i₂) :
    ctOut i₁ stride pad dk op ≤ ctOut i₂ stride pad dk op := by
  unfold ctOut
  have hm : (i₁ - 1) * stride ≤ (i₂ - 1) * stride :=
    Nat.mul_le_mul_right _ (Nat.sub_le_sub_right h 1)
  have hadd : (i₁ - 1) * stride + dk + op + 1 ≤ (i₂ - 1) * stride + dk + op + 1 := by
    exact Nat.add_le_add_right (Nat.add_le_add_right (Nat.add_le_add_right hm _) _) 1
  exact Nat.sub_le_sub_right hadd (2 * pad)

/-- **Upsampling lower bound**: with stride ≥ 1 and no padding, transpose conv
    never shrinks the input length. -/
theorem ctOut_ge (input stride dk op : Nat) (hs : 1 ≤ stride) (h : 1 ≤ input) :
    input ≤ ctOut input stride 0 dk op := by
  rw [ctOut_no_pad]
  have hm : (input - 1) * 1 ≤ (input - 1) * stride := Nat.mul_le_mul_left _ hs
  simp at hm
  omega

def ctShape (n cout lOut : Nat) : List Nat := [n, cout, lOut]

theorem ct_rank (n cout lOut : Nat) : (ctShape n cout lOut).length = 3 := rfl
theorem ct_batch (n cout lOut : Nat) : (ctShape n cout lOut).head? = some n := rfl
theorem ct_channels (n cout lOut : Nat) : (ctShape n cout lOut).get? 1 = some cout := rfl

end ConvTranspose
end TensorGuard
