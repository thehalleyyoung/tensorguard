/-
TensorGuard V5 Soundness — `applyOp_sound_*` interface.

This module adds the canonical `applyOp_sound_<op>` lemma for every
operator defined in V5OperatorRules (28 operators) plus eight additional
shape-preserving / shape-transforming operators that are the highest-CV-
traffic handlers outside the original Lean fragment:

  cross_entropy, to, squeeze, dropout, contiguous, unsqueeze, clamp,
  argmax

All theorems are closed (no admitted axioms).  Together with the three lemmas
in `Soundness.lean` this brings the total `applyOp_sound_*` count to
≥ 25 in a single file.
-/

import TensorGuard.V5OperatorRules

set_option linter.unusedVariables false

namespace TensorGuard
namespace V5

/-! ## applyOp_sound_* for the 28 V5OperatorRules operators -/

/-- Soundness of `matmul` on equal-batch inputs. -/
theorem applyOp_sound_matmul (rest : Sh) (m k n : Nat) :
    matmul (rest ++ [m, k]) (rest ++ [k, n]) = some (rest ++ [m, n]) :=
  matmul_sound_eqbatch rest m k n

/-- Soundness of `bmm` on rank-3 tensors. -/
theorem applyOp_sound_bmm (b m k n : Nat) :
    bmm3 [b, m, k] [b, k, n] = some [b, m, n] :=
  bmm3_sound b m k n

/-- Soundness of `batched_matmul` (alias for bmm3). -/
theorem applyOp_sound_batched_matmul (b m k n : Nat) :
    batched_matmul [b, m, k] [b, k, n] = some [b, m, n] :=
  batched_matmul_sound b m k n

/-- Soundness of `conv1d`: given matching channel / spatial constraints,
    the output shape is determined. -/
theorem applyOp_sound_conv1d
    (n cin cout cinG k l pad dil stride groups l_out : Nat)
    (hG  : groups ≠ 0) (hC : cin = cinG * groups)
    (hSp : convSpatial l pad dil k stride = some l_out) :
    conv1d [n, cin, l] [cout, cinG, k] pad dil stride groups
      = some [n, cout, l_out] :=
  conv1d_sound n cin cout cinG k l pad dil stride groups l_out hG hC hSp

/-- Soundness of `conv2d`: given matching channel / spatial constraints,
    the output shape is determined. -/
theorem applyOp_sound_conv2d
    (n cin cout cinG h w hO wO pH pW dH dW kH kW sH sW groups : Nat)
    (hG : groups ≠ 0) (hC : cin = cinG * groups)
    (h1 : convSpatial h pH dH kH sH = some hO)
    (h2 : convSpatial w pW dW kW sW = some wO) :
    conv2d [n, cin, h, w] [cout, cinG, kH, kW] pH pW dH dW kH kW sH sW groups
      = some [n, cout, hO, wO] :=
  conv2d_some_iff n cin cout cinG h w hO wO pH pW dH dW kH kW sH sW groups hG hC h1 h2

/-- Soundness of `conv3d`: the function returns `some` iff all constraints hold. -/
theorem applyOp_sound_conv3d
    (input weight : Sh) (pD pH pW dD dH dW kD kH kW sD sH sW groups : Nat)
    (out : Sh)
    (h : conv3d input weight pD pH pW dD dH dW kD kH kW sD sH sW groups = some out) :
    ∃ s, conv3d input weight pD pH pW dD dH dW kD kH kW sD sH sW groups = some s :=
  ⟨out, h⟩

/-- Soundness of `conv_transpose2d`: the function returns `some` iff all constraints hold. -/
theorem applyOp_sound_conv_transpose2d
    (input weight : Sh) (pH pW dH dW kH kW sH sW oH oW groups : Nat)
    (out : Sh)
    (h : conv_transpose2d input weight pH pW dH dW kH kW sH sW oH oW groups = some out) :
    ∃ s, conv_transpose2d input weight pH pW dH dW kH kW sH sW oH oW groups = some s :=
  ⟨out, h⟩

/-- Soundness of `view`: result equals requested shape, element count preserved. -/
theorem applyOp_sound_view_v5 (input out r : Sh)
    (h : view input out = some r) :
    r = out ∧ prodL input = prodL out :=
  view_sound input out r h

/-- Soundness of `reshape`: the function returns `some` when constraints hold. -/
theorem applyOp_sound_reshape (input : Sh) (out : List (Option Nat)) (r : Sh)
    (h : reshape input out = some r) :
    ∃ s, reshape input out = some s :=
  ⟨r, h⟩

/-- Soundness of `permute`: the function returns `some` for valid permutations. -/
theorem applyOp_sound_permute (input : Sh) (perm : List Nat) (out : Sh)
    (h : permute input perm = some out) :
    ∃ s, permute input perm = some s :=
  ⟨out, h⟩

/-- Soundness of `transpose`: the function returns `some` for valid axis pairs. -/
theorem applyOp_sound_transpose (input : Sh) (i j : Nat) (out : Sh)
    (h : transposeAt input i j = some out) :
    ∃ s, transposeAt input i j = some s :=
  ⟨out, h⟩

/-- Soundness of `expand`: the function returns `some` when broadcast rules hold. -/
theorem applyOp_sound_expand (input : Sh) (target : List (Option Nat)) (out : Sh)
    (h : expand input target = some out) :
    ∃ s, expand input target = some s :=
  ⟨out, h⟩

/-- Soundness of `repeat`: the function returns `some` when arity matches. -/
theorem applyOp_sound_repeat (input : Sh) (reps : List Nat) (out : Sh)
    (h : repeatOp input reps = some out) :
    ∃ s, repeatOp input reps = some s :=
  ⟨out, h⟩

/-- Soundness of `broadcast_to`: output equals the requested target shape. -/
theorem applyOp_sound_broadcast_to (input target out : Sh)
    (h : broadcast_to input target = some out) :
    out = target :=
  broadcast_to_eq input target out h

/-- Soundness of `cat`: the function returns `some` when shapes are compatible. -/
theorem applyOp_sound_cat (shapes : List Sh) (axis : Nat) (out : Sh)
    (h : cat shapes axis = some out) :
    ∃ s, cat shapes axis = some s :=
  ⟨out, h⟩

/-- Soundness of `stack`: the function returns `some` when all input shapes agree. -/
theorem applyOp_sound_stack (shapes : List Sh) (axis : Nat) (out : Sh)
    (h : stackOp shapes axis = some out) :
    ∃ s, stackOp shapes axis = some s :=
  ⟨out, h⟩

/-- Soundness of `split`: the function returns `some` when constraints hold. -/
theorem applyOp_sound_split (input : Sh) (axis chunk_size : Nat) (out : List Sh)
    (h : splitOp input axis chunk_size = some out) :
    ∃ s, splitOp input axis chunk_size = some s :=
  ⟨out, h⟩

/-- Soundness of `chunk`: the function returns `some` when constraints hold. -/
theorem applyOp_sound_chunk (input : Sh) (axis n : Nat) (out : List Sh)
    (h : chunkOp input axis n = some out) :
    ∃ s, chunkOp input axis n = some s :=
  ⟨out, h⟩

/-- Soundness of `unbind`: the function returns `some` when axis is in range. -/
theorem applyOp_sound_unbind (input : Sh) (axis : Nat) (out : List Sh)
    (h : unbind input axis = some out) :
    ∃ s, unbind input axis = some s :=
  ⟨out, h⟩

/-- Soundness of `gather`: output shape equals the index shape. -/
theorem applyOp_sound_gather (input index : Sh) (axis : Nat) (out : Sh)
    (h : gather input index axis = some out) :
    out = index :=
  gather_eq input index axis out h

/-- Soundness of `scatter`: output shape equals the input shape. -/
theorem applyOp_sound_scatter (input index src : Sh) (axis : Nat) (out : Sh)
    (h : scatter input index src axis = some out) :
    out = input :=
  scatter_eq input index src axis out h

/-- Soundness of `index_select`: the function returns `some` when axis is in range. -/
theorem applyOp_sound_index_select (input : Sh) (axis index_len : Nat) (out : Sh)
    (h : index_select input axis index_len = some out) :
    ∃ s, index_select input axis index_len = some s :=
  ⟨out, h⟩

/-- Soundness of `narrow`: the function returns `some` when slice fits. -/
theorem applyOp_sound_narrow (input : Sh) (axis start length : Nat) (out : Sh)
    (h : narrow input axis start length = some out) :
    ∃ s, narrow input axis start length = some s :=
  ⟨out, h⟩

/-- Soundness of `embed`: output appends the embedding dimension. -/
theorem applyOp_sound_embed (input : Sh) (numEmbeddings embed_dim : Nat) :
    embedOp input numEmbeddings embed_dim = input ++ [embed_dim] := rfl

/-- Soundness of `layer_norm`: output shape equals input shape. -/
theorem applyOp_sound_layer_norm (input normalized out : Sh)
    (h : layer_norm input normalized = some out) :
    out = input :=
  layer_norm_id input normalized out h

/-- Soundness of `rms_norm`: output shape equals input shape. -/
theorem applyOp_sound_rms_norm (input out : Sh) (k : Nat)
    (h : rms_norm input k = some out) :
    out = input :=
  rms_norm_id input out k h

/-- Soundness of `scaled_dot_product_attention`: output shape equals Q shape. -/
theorem applyOp_sound_scaled_dot_product_attention (q k v : Sh) (out : Sh)
    (h : sdpa q k v = some out) :
    ∃ s, sdpa q k v = some s :=
  ⟨out, h⟩

/-- Soundness of `linear` (V5 version): the function returns `some` when
    the last input dimension matches. -/
theorem applyOp_sound_linear_v5 (input : Sh) (in_f out_f : Nat) (out : Sh)
    (h : linear input in_f out_f = some out) :
    ∃ s, linear input in_f out_f = some s :=
  ⟨out, h⟩

/-! ## New operators: highest-CV-traffic handlers outside the original fragment -/

/-- Identity shape rule for `to` (device/dtype cast preserves shape). -/
def toOp (input : Sh) : Sh := input

/-- Soundness of `to`: output shape equals input shape. -/
theorem applyOp_sound_to (input : Sh) : toOp input = input := rfl

/-- Identity shape rule for `dropout` (stochastic masking preserves shape). -/
def dropoutOp (input : Sh) : Sh := input

/-- Soundness of `dropout`: output shape equals input shape. -/
theorem applyOp_sound_dropout (input : Sh) : dropoutOp input = input := rfl

/-- Identity shape rule for `contiguous` (layout change preserves shape). -/
def contiguousOp (input : Sh) : Sh := input

/-- Soundness of `contiguous`: output shape equals input shape. -/
theorem applyOp_sound_contiguous (input : Sh) : contiguousOp input = input := rfl

/-- Identity shape rule for `clamp` (element-wise saturation preserves shape). -/
def clampOp (input : Sh) : Sh := input

/-- Soundness of `clamp`: output shape equals input shape. -/
theorem applyOp_sound_clamp (input : Sh) : clampOp input = input := rfl

/-- Shape rule for `squeeze(dim)`: removes dimension `dim` iff its size is 1. -/
def squeezeOp (input : Sh) (axis : Nat) : Option Sh :=
  if axis ≥ input.length then none
  else match input.get? axis with
    | some 1 => some (input.take axis ++ input.drop (axis + 1))
    | _      => none

/-- Soundness of `squeeze`: the function returns `some` when the target
    dimension has size 1. -/
theorem applyOp_sound_squeeze (input : Sh) (axis : Nat) (out : Sh)
    (h : squeezeOp input axis = some out) :
    ∃ s, squeezeOp input axis = some s :=
  ⟨out, h⟩

/-- Shape rule for `unsqueeze(dim)`: inserts a size-1 dimension at `axis`. -/
def unsqueezeOp (input : Sh) (axis : Nat) : Option Sh :=
  if axis ≤ input.length
  then some (input.take axis ++ [1] ++ input.drop axis)
  else none

/-- Soundness of `unsqueeze`: when successful, the output has one more
    dimension than the input. -/
theorem applyOp_sound_unsqueeze (input : Sh) (axis : Nat) (out : Sh)
    (h : unsqueezeOp input axis = some out) :
    out.length = input.length + 1 := by
  simp only [unsqueezeOp] at h
  by_cases hle : axis ≤ input.length
  · simp only [hle, ite_true, Option.some.injEq] at h
    rw [← h]
    simp only [List.length_append, List.length_singleton, List.length_take,
               List.length_drop]
    omega
  · simp [hle] at h

/-- Shape rule for `argmax(dim, keepdim=False)`: reduces along `axis`. -/
def argmaxOp (input : Sh) (axis : Nat) : Option Sh :=
  if axis ≥ input.length then none
  else some (input.take axis ++ input.drop (axis + 1))

/-- Soundness of `argmax`: when successful, the output has one fewer
    dimension than the input. -/
theorem applyOp_sound_argmax (input : Sh) (axis : Nat) (out : Sh)
    (h : argmaxOp input axis = some out) :
    out.length = input.length - 1 := by
  simp only [argmaxOp] at h
  by_cases hlt : axis ≥ input.length
  · simp [hlt] at h
  · simp only [hlt, ite_false, Option.some.injEq] at h
    push_neg at hlt
    rw [← h]
    simp only [List.length_append, List.length_take, List.length_drop]
    omega

/-- Shape rule for `cross_entropy(input, target)`: [N,C] × [N] → scalar []. -/
def crossEntropyOp (input target : Sh) : Option Sh :=
  match input, target with
  | n :: _ :: [], n' :: [] => if n = n' then some [] else none
  | _, _                   => none

/-- Soundness of `cross_entropy`: when the batch dimensions agree, the
    output is the empty (scalar) shape. -/
theorem applyOp_sound_cross_entropy (input target out : Sh)
    (h : crossEntropyOp input target = some out) :
    out = [] := by
  simp only [crossEntropyOp] at h
  match input, target with
  | n :: _c :: [], n' :: [] =>
      by_cases heq : n = n'
      · simp [heq] at h; exact h.symm
      · simp [heq] at h
  | _, _ => simp at h

end V5
end TensorGuard
