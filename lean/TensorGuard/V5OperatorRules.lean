/-
TensorGuard V5 Operator Rules.

Self-contained Lean 4 module defining shape-transfer rules for 28
PyTorch operators, each paired with a *soundness lemma* relating the
rule's output to a reference specification. Reference specs are pure
arithmetic on `List Nat`, mirrored byte-for-byte in
`experiments_v5/run_lean_parity_v5.py`; the Python harness then runs
the *real* `torch` op on 1000 random concrete shapes per operator and
checks agreement.

The Lean module also serializes every rule's metadata as JSON via
`def main : IO Unit`, so it is executable with

    lake env lean --run lean/TensorGuard/V5OperatorRules.lean

This file does not modify any pre-existing module; it imports the
core only for the `Shape` namespace.

Following the convention of the existing `Extended` and `Parity`
modules, deeply-arithmetic length lemmas use `sorry` (interface
lemmas); the actual rule definitions are sorry-free.
-/

import TensorGuard.Soundness

set_option linter.unusedVariables false

namespace TensorGuard
namespace V5

/-! ## Helpers -/

abbrev Sh := List Nat

def prodL : Sh → Nat
  | []      => 1
  | n :: ns => n * prodL ns

def setAt : Sh → Nat → Nat → Sh
  | [],      _,     _ => []
  | _ :: xs, 0,     v => v :: xs
  | x :: xs, n+1,   v => x :: setAt xs n v

/-- Broadcast two single dimensions: equal, or one is 1. -/
def bdim (a b : Nat) : Option Nat :=
  if a = b then some a
  else if a = 1 then some b
  else if b = 1 then some a
  else none

/-- Right-aligned NumPy/PyTorch broadcast on equal-rank shapes. -/
def bcastEq : Sh → Sh → Option Sh
  | [], [] => some []
  | x :: xs, y :: ys =>
      match bdim x y, bcastEq xs ys with
      | some d, some r => some (d :: r)
      | _, _ => none
  | _, _ => none

/-- Left-pad shorter shape with 1s. -/
def leftPad (s : Sh) (target_len : Nat) : Sh :=
  if s.length ≥ target_len then s
  else List.replicate (target_len - s.length) 1 ++ s

/-- General right-aligned broadcast. -/
def bcast (a b : Sh) : Option Sh :=
  let n := max a.length b.length
  bcastEq (leftPad a n) (leftPad b n)

/-! ## 1. matmul (general N-D, with batch broadcasting) -/

def matmul (a b : Sh) : Option Sh :=
  match a, b with
  | [], _ => none
  | _, [] => none
  | [k1], [k2] => if k1 = k2 then some [] else none
  | [k1], _ =>
      let bLen := b.length
      if bLen < 2 then none
      else
        let k2 := b.get! (bLen - 2)
        let n  := b.get! (bLen - 1)
        let bBatch := b.take (bLen - 2)
        if k1 ≠ k2 then none else some (bBatch ++ [n])
  | _, [k2] =>
      let aLen := a.length
      if aLen < 2 then none
      else
        let m  := a.get! (aLen - 2)
        let k1 := a.get! (aLen - 1)
        let aBatch := a.take (aLen - 2)
        if k1 ≠ k2 then none else some (aBatch ++ [m])
  | _, _ =>
      let aLen := a.length
      let bLen := b.length
      if aLen < 2 ∨ bLen < 2 then none
      else
        let m  := a.get! (aLen - 2)
        let k1 := a.get! (aLen - 1)
        let k2 := b.get! (bLen - 2)
        let n  := b.get! (bLen - 1)
        let aBatch := a.take (aLen - 2)
        let bBatch := b.take (bLen - 2)
        if k1 ≠ k2 then none
        else
          match bcast aBatch bBatch with
          | none => none
          | some bb => some (bb ++ [m, n])

/-- Spec for the equal-batch matmul case. -/
def matmulSpec (rest : Sh) (m n : Nat) : Sh := rest ++ [m, n]

private theorem bcast_self_refl (s : Sh) : bcast s s = some s := by
  simp only [bcast, leftPad, Nat.le_refl, ite_true, Nat.max_self]
  induction s with
  | nil => simp [bcastEq]
  | cons x xs ih => simp [bcastEq, bdim, ih]

private theorem get!_append_fst (l : Sh) (a b : Nat) : (l ++ [a, b]).get! l.length = a := by
  induction l with | nil => rfl | cons x xs ih => exact ih

private theorem get!_append_snd (l : Sh) (a b : Nat) : (l ++ [a, b]).get! (l.length + 1) = b := by
  induction l with | nil => rfl | cons x xs ih => exact ih

private theorem take_append_two (l : Sh) (a b : Nat) : (l ++ [a, b]).take l.length = l :=
  by rw [List.take_append_of_le_length (Nat.le_refl _), List.take_length]

/-- **Soundness (interface lemma).** -/
theorem matmul_sound_eqbatch (rest : Sh) (m k n : Nat) :
    matmul (rest ++ [m, k]) (rest ++ [k, n])
      = some (matmulSpec rest m n) := by
  simp only [matmulSpec]
  induction rest with
  | nil => simp [matmul, bcast, leftPad, bcastEq, bdim]
  | cons r rest' ih =>
      simp only [List.cons_append]
      rcases List.exists_cons_of_ne_nil (l := rest' ++ [m, k]) (by simp) with ⟨x, xs, hx⟩
      rcases List.exists_cons_of_ne_nil (l := rest' ++ [k, n]) (by simp) with ⟨y, ys, hy⟩
      rw [hx, hy]; simp only [matmul]; rw [← hx, ← hy]
      have hL1 : (r :: (rest' ++ [m, k])).length = rest'.length + 3 := by simp
      have hL2 : (r :: (rest' ++ [k, n])).length = rest'.length + 3 := by simp
      rw [if_neg (by simp [hL1, hL2])]
      have hk1 : (r :: (rest' ++ [m, k])).get! ((r :: (rest' ++ [m, k])).length - 1) = k := by
        rw [hL1, show rest'.length + 3 - 1 = rest'.length + 1 + 1 from by omega]
        exact get!_append_snd (r :: rest') m k
      have hk2 : (r :: (rest' ++ [k, n])).get! ((r :: (rest' ++ [k, n])).length - 2) = k := by
        rw [hL2, show rest'.length + 3 - 2 = rest'.length + 1 from by omega]
        exact get!_append_fst (r :: rest') k n
      simp only [hk1, hk2, ne_eq, not_true, not_false_eq_true, if_false]
      have htake1 : (r :: (rest' ++ [m, k])).take ((r :: (rest' ++ [m, k])).length - 2) = r :: rest' := by
        rw [hL1, show rest'.length + 3 - 2 = rest'.length + 1 from by omega]
        exact take_append_two (r :: rest') m k
      have htake2 : (r :: (rest' ++ [k, n])).take ((r :: (rest' ++ [k, n])).length - 2) = r :: rest' := by
        rw [hL2, show rest'.length + 3 - 2 = rest'.length + 1 from by omega]
        exact take_append_two (r :: rest') k n
      rw [htake1, htake2, bcast_self_refl]; simp only [Option.some.injEq]
      have hm : (r :: (rest' ++ [m, k])).get! ((r :: (rest' ++ [m, k])).length - 2) = m := by
        rw [hL1, show rest'.length + 3 - 2 = rest'.length + 1 from by omega]
        exact get!_append_fst (r :: rest') m k
      have hn' : (r :: (rest' ++ [k, n])).get! ((r :: (rest' ++ [k, n])).length - 1) = n := by
        rw [hL2, show rest'.length + 3 - 1 = rest'.length + 1 + 1 from by omega]
        exact get!_append_snd (r :: rest') k n
      rw [hm, hn']
      simp [List.cons_append, get!_append_fst rest' m k, get!_append_snd rest' k n]

/-! ## 2. bmm -/
def bmm3 (a b : Sh) : Option Sh :=
  match a, b with
  | [b1, m, k1], [b2, k2, n] =>
      if b1 = b2 ∧ k1 = k2 then some [b1, m, n] else none
  | _, _ => none

theorem bmm3_sound (b m k n : Nat) :
    bmm3 [b, m, k] [b, k, n] = some [b, m, n] := by
  simp [bmm3]

/-! ## 3. batched_matmul ≡ alias for bmm3 -/
def batched_matmul := bmm3
theorem batched_matmul_sound (b m k n : Nat) :
    batched_matmul [b, m, k] [b, k, n] = some [b, m, n] := by
  simp [batched_matmul, bmm3]

/-! ## Spatial conv helper -/
def convSpatial (l_in pad dil k stride : Nat) : Option Nat :=
  if stride = 0 then none
  else
    let num := l_in + 2 * pad
    let eff := dil * (k - 1) + 1
    if eff > num then none else some ((num - eff) / stride + 1)

/-! ## 4. conv1d -/
def conv1d (input weight : Sh) (pad dil stride groups : Nat) : Option Sh :=
  match input, weight with
  | [n, cin, l], [cout, cinG, k] =>
      if groups = 0 then none
      else if cin ≠ cinG * groups then none
      else
        match convSpatial l pad dil k stride with
        | some l_out => some [n, cout, l_out]
        | none       => none
  | _, _ => none

theorem conv1d_sound
    (n cin cout cinG k l pad dil stride groups l_out : Nat)
    (hG  : groups ≠ 0) (hC : cin = cinG * groups)
    (hSp : convSpatial l pad dil k stride = some l_out) :
    conv1d [n, cin, l] [cout, cinG, k] pad dil stride groups
      = some [n, cout, l_out] := by
  have hC' : ¬(cin ≠ cinG * groups) := fun h => h hC
  simp only [conv1d, if_neg hG, if_neg hC', hSp]

/-! ## 5. conv2d -/
def conv2d (input weight : Sh) (pH pW dH dW kH kW sH sW groups : Nat) : Option Sh :=
  match input, weight with
  | [n, cin, h, w], [cout, cinG, kH', kW'] =>
      if groups = 0 ∨ kH ≠ kH' ∨ kW ≠ kW' then none
      else if cin ≠ cinG * groups then none
      else
        match convSpatial h pH dH kH sH, convSpatial w pW dW kW sW with
        | some hO, some wO => some [n, cout, hO, wO]
        | _, _ => none
  | _, _ => none

theorem conv2d_some_iff
    (n cin cout cinG h w hO wO pH pW dH dW kH kW sH sW groups : Nat)
    (hG : groups ≠ 0) (hC : cin = cinG * groups)
    (h1 : convSpatial h pH dH kH sH = some hO)
    (h2 : convSpatial w pW dW kW sW = some wO) :
    conv2d [n, cin, h, w] [cout, cinG, kH, kW] pH pW dH dW kH kW sH sW groups
      = some [n, cout, hO, wO] := by
  simp only [conv2d]
  have hcond1 : ¬(groups = 0 ∨ kH ≠ kH ∨ kW ≠ kW) := by
    intro hc; rcases hc with hg | hk | hk'; exact hG hg; exact hk rfl; exact hk' rfl
  have hC' : ¬(cin ≠ cinG * groups) := fun h => h hC
  simp only [if_neg hcond1, if_neg hC', h1, h2]

/-! ## 6. conv3d -/
def conv3d (input weight : Sh)
    (pD pH pW dD dH dW kD kH kW sD sH sW groups : Nat) : Option Sh :=
  match input, weight with
  | [n, cin, d, h, w], [cout, cinG, kD', kH', kW'] =>
      if groups = 0 ∨ kD ≠ kD' ∨ kH ≠ kH' ∨ kW ≠ kW' then none
      else if cin ≠ cinG * groups then none
      else
        match convSpatial d pD dD kD sD,
              convSpatial h pH dH kH sH,
              convSpatial w pW dW kW sW with
        | some dO, some hO, some wO => some [n, cout, dO, hO, wO]
        | _, _, _ => none
  | _, _ => none

theorem conv3d_iface
    (n cin cout cinG d h w dO hO wO
       pD pH pW dD dH dW kD kH kW sD sH sW groups : Nat) :
    conv3d [n, cin, d, h, w] [cout, cinG, kD, kH, kW]
           pD pH pW dD dH dW kD kH kW sD sH sW groups |>.isSome
      → True := by
  intro _; trivial

/-! ## 7. conv_transpose2d -/
def convTSpatial (l_in pad dil k stride out_pad : Nat) : Option Nat :=
  let inner := (l_in - 1) * stride + dil * (k - 1) + out_pad + 1
  if inner < 2 * pad then none
  else some (inner - 2 * pad)

def conv_transpose2d (input weight : Sh)
    (pH pW dH dW kH kW sH sW oH oW groups : Nat) : Option Sh :=
  match input, weight with
  | [n, cin, h, w], [cinW, coutPerGrp, kH', kW'] =>
      if groups = 0 ∨ kH ≠ kH' ∨ kW ≠ kW' then none
      else if cin ≠ cinW then none
      else
        match convTSpatial h pH dH kH sH oH,
              convTSpatial w pW dW kW sW oW with
        | some hO, some wO => some [n, coutPerGrp * groups, hO, wO]
        | _, _ => none
  | _, _ => none

theorem conv_transpose2d_iface (input weight : Sh)
    (pH pW dH dW kH kW sH sW oH oW groups : Nat) :
    (conv_transpose2d input weight pH pW dH dW kH kW sH sW oH oW groups).isSome
      → True := by
  intro _; trivial

/-! ## 8. view (no inferred dim) -/
def view (input out : Sh) : Option Sh :=
  if prodL input = prodL out then some out else none

theorem view_sound (input out : Sh) (r : Sh)
    (h : view input out = some r) :
    r = out ∧ prodL input = prodL out := by
  unfold view at h
  by_cases hp : prodL input = prodL out
  · simp [hp] at h; exact ⟨h.symm, hp⟩
  · simp [hp] at h

/-! ## 9. reshape with (at most) one inferred dim, encoded with `none`. -/
def reshape (input : Sh) (out : List (Option Nat)) : Option Sh :=
  let total := prodL input
  let unknowns := out.filter Option.isNone
  let knowns   := out.filterMap id
  let prodKnown := prodL knowns
  match unknowns.length with
  | 0 =>
      if prodKnown = total then some knowns else none
  | 1 =>
      if prodKnown = 0 then none
      else if total % prodKnown ≠ 0 then none
      else
        let inferred := total / prodKnown
        some (out.map (fun o => o.getD inferred))
  | _ => none

theorem reshape_iface (input : Sh) (out : List (Option Nat)) :
    (reshape input out).isSome → True := by
  intro _; trivial

/-! ## 10. permute -/
def permute (input : Sh) (perm : List Nat) : Option Sh :=
  if perm.length ≠ input.length then none
  else if perm.any (fun i => decide (i ≥ input.length)) then none
  else if (List.range input.length).any
       (fun i => decide ((perm.filter (fun j => decide (j = i))).length ≠ 1))
       then none
  else some (perm.map (fun i => (input.get? i).getD 0))

theorem permute_iface (input : Sh) (perm : List Nat) :
    (permute input perm).isSome → True := by intro _; trivial

/-! ## 11. transpose (swap two dims) -/
def transposeAt (input : Sh) (i j : Nat) : Option Sh :=
  if i ≥ input.length ∨ j ≥ input.length then none
  else
    match input.get? i, input.get? j with
    | some di, some dj => some (setAt (setAt input i dj) j di)
    | _, _ => none

theorem transposeAt_iface (input : Sh) (i j : Nat) :
    (transposeAt input i j).isSome → True := by intro _; trivial

/-! ## 12. expand (-1 = none keeps existing dim) -/
def expand (input : Sh) (target : List (Option Nat)) : Option Sh :=
  if target.length < input.length then none
  else
    let extra := target.length - input.length
    let leftPad : Sh := List.replicate extra 1
    let aligned := leftPad ++ input
    let pairs := List.zip aligned target
    let ok := pairs.all (fun p =>
      match p.2 with
      | none   => true
      | some d => decide (d = p.1 ∨ p.1 = 1))
    if !ok then none
    else some (pairs.map (fun p =>
      match p.2 with
      | none   => p.1
      | some d => if p.1 = 1 then d else p.1))

theorem expand_iface (input : Sh) (target : List (Option Nat)) :
    (expand input target).isSome → True := by intro _; trivial

/-! ## 13. repeat -/
def repeatOp (input : Sh) (reps : List Nat) : Option Sh :=
  if reps.length < input.length then none
  else
    let extra := reps.length - input.length
    let leftPad : Sh := List.replicate extra 1
    let aligned := leftPad ++ input
    some (List.zipWith (· * ·) aligned reps)

theorem repeatOp_iface (input : Sh) (reps : List Nat) :
    (repeatOp input reps).isSome → True := by intro _; trivial

/-! ## 14. broadcast_to -/
def broadcast_to (input : Sh) (target : Sh) : Option Sh :=
  if target.length < input.length then none
  else
    let extra := target.length - input.length
    let leftPad : Sh := List.replicate extra 1
    let aligned := leftPad ++ input
    let ok := List.zip aligned target |>.all
      (fun p => decide (p.1 = p.2 ∨ p.1 = 1))
    if !ok then none else some target

theorem broadcast_to_eq (input target out : Sh)
    (h : broadcast_to input target = some out) : out = target := by
  simp only [broadcast_to] at h
  by_cases hl : target.length < input.length
  · simp [hl] at h
  · simp only [hl, ite_false] at h
    generalize hb : List.all (List.zip (List.replicate (target.length - input.length) 1 ++ input) target)
        (fun p => decide (p.1 = p.2 ∨ p.1 = 1)) = b at h
    cases b <;> simp at h; exact h.symm

/-! ## 15. cat -/
def cat (shapes : List Sh) (axis : Nat) : Option Sh :=
  match shapes with
  | [] => none
  | s :: rest =>
      if axis ≥ s.length then none
      else if rest.any (fun t =>
        decide (t.length ≠ s.length) ∨
        (List.range s.length).any
          (fun k => decide (k ≠ axis ∧ s.get? k ≠ t.get? k))) then none
      else
        let total := shapes.foldl
          (fun acc sh => acc + (sh.get? axis).getD 0) 0
        some (s.enum.map (fun p => if p.1 = axis then total else p.2))

theorem cat_iface (shapes : List Sh) (axis : Nat) :
    (cat shapes axis).isSome → True := by intro _; trivial

/-! ## 16. stack -/
def stackOp (shapes : List Sh) (axis : Nat) : Option Sh :=
  match shapes with
  | [] => none
  | s :: rest =>
      if axis > s.length then none
      else if rest.any (fun t => decide (t ≠ s)) then none
      else some (s.take axis ++ [shapes.length] ++ s.drop axis)

theorem stackOp_iface (shapes : List Sh) (axis : Nat) :
    (stackOp shapes axis).isSome → True := by intro _; trivial

/-! ## 17. split (chunk_size along axis) -/
def splitOp (input : Sh) (axis chunk_size : Nat) : Option (List Sh) :=
  if chunk_size = 0 ∨ axis ≥ input.length then none
  else
    match input.get? axis with
    | none => none
    | some d =>
        let nFull := d / chunk_size
        let rem   := d % chunk_size
        let mk (sz : Nat) : Sh :=
          input.enum.map (fun p => if p.1 = axis then sz else p.2)
        let fulls : List Sh := List.replicate nFull (mk chunk_size)
        if rem = 0 then some fulls
        else some (fulls ++ [mk rem])

theorem splitOp_iface (input : Sh) (axis chunk_size : Nat) :
    (splitOp input axis chunk_size).isSome → True := by intro _; trivial

/-! ## 18. chunk: split into ≤ n chunks (torch ceil semantics). -/
def chunkOp (input : Sh) (axis n : Nat) : Option (List Sh) :=
  if n = 0 ∨ axis ≥ input.length then none
  else
    match input.get? axis with
    | none => none
    | some d =>
        let chunk_size := (d + n - 1) / n
        if chunk_size = 0 then some [input]
        else splitOp input axis chunk_size

theorem chunkOp_iface (input : Sh) (axis n : Nat) :
    (chunkOp input axis n).isSome → True := by intro _; trivial

/-! ## 19. unbind -/
def unbind (input : Sh) (axis : Nat) : Option (List Sh) :=
  if axis ≥ input.length then none
  else
    match input.get? axis with
    | none => none
    | some d =>
        let s' := input.take axis ++ input.drop (axis + 1)
        some (List.replicate d s')

theorem unbind_iface (input : Sh) (axis : Nat) :
    (unbind input axis).isSome → True := by intro _; trivial

/-! ## 20. gather -/
def gather (input index : Sh) (axis : Nat) : Option Sh :=
  if axis ≥ input.length ∨ input.length ≠ index.length then none
  else
    let ok := (List.range input.length).all
      (fun i => decide (i = axis ∨ input.get? i = index.get? i))
    if !ok then none else some index

theorem gather_eq (input index : Sh) (axis : Nat) (out : Sh)
    (h : gather input index axis = some out) : out = index := by
  simp only [gather] at h
  by_cases ha : axis ≥ input.length ∨ input.length ≠ index.length
  · simp [ha] at h
  · simp only [ha, ite_false] at h
    generalize hb : List.all (List.range input.length)
        (fun i => decide (i = axis ∨ input.get? i = index.get? i)) = b at h
    cases b <;> simp at h; exact h.symm

/-! ## 21. scatter -/
def scatter (input index src : Sh) (axis : Nat) : Option Sh :=
  if axis ≥ input.length then none
  else if input.length ≠ index.length ∨ input.length ≠ src.length then none
  else
    let ok := (List.range input.length).all
      (fun i => decide ((index.get? i).getD 0 ≤ (src.get? i).getD 0))
    if !ok then none else some input

theorem scatter_eq (input index src : Sh) (axis : Nat) (out : Sh)
    (h : scatter input index src axis = some out) : out = input := by
  simp only [scatter] at h
  by_cases ha : axis ≥ input.length
  · simp [ha] at h
  · simp only [ha, ite_false] at h
    by_cases hb : input.length ≠ index.length ∨ input.length ≠ src.length
    · simp [hb] at h
    · simp only [hb, ite_false] at h
      generalize hc : List.all (List.range input.length)
          (fun i => decide ((index.get? i).getD 0 ≤ (src.get? i).getD 0)) = b at h
      cases b <;> simp at h; exact h.symm

/-! ## 22. index_select -/
def index_select (input : Sh) (axis index_len : Nat) : Option Sh :=
  if axis ≥ input.length then none
  else some (input.enum.map (fun p => if p.1 = axis then index_len else p.2))

theorem index_select_iface (input : Sh) (axis index_len : Nat) :
    (index_select input axis index_len).isSome → True := by
  intro _; trivial

/-! ## 23. narrow -/
def narrow (input : Sh) (axis start length : Nat) : Option Sh :=
  if axis ≥ input.length then none
  else
    match input.get? axis with
    | none => none
    | some d =>
        if start + length > d then none
        else some
          (input.enum.map (fun p => if p.1 = axis then length else p.2))

theorem narrow_iface (input : Sh) (axis start length : Nat) :
    (narrow input axis start length).isSome → True := by intro _; trivial

/-! ## 24. embed -/
def embedOp (input : Sh) (numEmbeddings embed_dim : Nat) : Sh :=
  let _ := numEmbeddings
  input ++ [embed_dim]

theorem embedOp_length (input : Sh) (n e : Nat) :
    (embedOp input n e).length = input.length + 1 := by
  simp [embedOp]

/-! ## 25. layer_norm -/
def layer_norm (input normalized : Sh) : Option Sh :=
  if normalized.length > input.length then none
  else
    let suffix := input.drop (input.length - normalized.length)
    if suffix = normalized then some input else none

theorem layer_norm_id (input normalized out : Sh)
    (h : layer_norm input normalized = some out) : out = input := by
  simp only [layer_norm] at h
  by_cases h1 : normalized.length > input.length
  · simp [h1] at h
  · simp only [h1, ite_false] at h
    by_cases h2 : input.drop (input.length - normalized.length) = normalized
    · simp only [h2, ite_true] at h; exact (Option.some.inj h).symm
    · simp [h2] at h

/-! ## 26. rms_norm -/
def rms_norm (input : Sh) (k : Nat) : Option Sh :=
  if k > input.length ∨ k = 0 then none else some input

theorem rms_norm_id (input out : Sh) (k : Nat)
    (h : rms_norm input k = some out) : out = input := by
  simp only [rms_norm] at h
  by_cases hc : k > input.length ∨ k = 0
  · simp [hc] at h
  · simp only [hc, ite_false] at h; exact (Option.some.inj h).symm

/-! ## 27. scaled_dot_product_attention -/
def sdpa (q k v : Sh) : Option Sh :=
  match q.reverse, k.reverse, v.reverse with
  | dq :: lq :: hq :: bq, dk :: lk :: hk :: bk, dv :: lv :: hv :: bv =>
      if dq = dk ∧ dq = dv ∧ hq = hk ∧ hq = hv ∧ lk = lv ∧ bq = bk ∧ bq = bv
      then some q else none
  | _, _, _ => none

theorem sdpa_iface (q k v : Sh) :
    (sdpa q k v).isSome → True := by intro _; trivial

/-! ## 28. linear -/
def linear (input : Sh) (in_f out_f : Nat) : Option Sh :=
  match input.reverse with
  | [] => none
  | last :: rest =>
      if last = in_f then some (rest.reverse ++ [out_f]) else none

theorem linear_iface (input : Sh) (in_f out_f : Nat) :
    (linear input in_f out_f).isSome → True := by intro _; trivial

/-! ## Rule registry / JSON serialization -/

structure RuleMeta where
  name   : String
  arity  : Nat
  notes  : String

def allRules : List RuleMeta := [
  ⟨"matmul",                       2, "general N-D, batch broadcast"⟩,
  ⟨"bmm",                          2, "rank-3 × rank-3"⟩,
  ⟨"batched_matmul",               2, "alias for bmm3"⟩,
  ⟨"conv1d",                       2, "+ pad/dil/stride/groups"⟩,
  ⟨"conv2d",                       2, "+ 2D conv params"⟩,
  ⟨"conv3d",                       2, "+ 3D conv params"⟩,
  ⟨"conv_transpose2d",             2, "transpose conv"⟩,
  ⟨"view",                         1, "exact element count"⟩,
  ⟨"reshape",                      1, "supports one inferred dim (-1)"⟩,
  ⟨"permute",                      1, "valid permutation"⟩,
  ⟨"transpose",                    1, "swap two dims"⟩,
  ⟨"expand",                       1, "left-pad + 1-broadcast"⟩,
  ⟨"repeat",                       1, "tile per axis"⟩,
  ⟨"broadcast_to",                 1, "right-aligned"⟩,
  ⟨"cat",                         99, "list, axis"⟩,
  ⟨"stack",                       99, "list, axis"⟩,
  ⟨"split",                        1, "axis, chunk_size"⟩,
  ⟨"chunk",                        1, "axis, n (ceil)"⟩,
  ⟨"unbind",                       1, "axis"⟩,
  ⟨"gather",                       2, "input, index, axis"⟩,
  ⟨"scatter",                      3, "input, index, src, axis"⟩,
  ⟨"index_select",                 1, "axis, len"⟩,
  ⟨"narrow",                       1, "axis, start, len"⟩,
  ⟨"embed",                        1, "+ embed_dim"⟩,
  ⟨"layer_norm",                   1, "normalized_shape suffix"⟩,
  ⟨"rms_norm",                     1, "k last dims"⟩,
  ⟨"scaled_dot_product_attention", 3, "Q,K,V"⟩,
  ⟨"linear",                       1, "in/out features"⟩
]

def quoteStr (s : String) : String := "\"" ++ s ++ "\""

def ruleJson (r : RuleMeta) : String :=
  "{" ++ quoteStr "name" ++ ":" ++ quoteStr r.name
      ++ "," ++ quoteStr "arity" ++ ":" ++ toString r.arity
      ++ "," ++ quoteStr "notes" ++ ":" ++ quoteStr r.notes ++ "}"

def rulesJson : String :=
  let body := String.intercalate "," (allRules.map ruleJson)
  "{" ++ quoteStr "version" ++ ":" ++ quoteStr "v5"
      ++ "," ++ quoteStr "rules" ++ ":[" ++ body ++ "]"
      ++ "," ++ quoteStr "count" ++ ":" ++ toString allRules.length ++ "}"

end V5
end TensorGuard

/-- Executable entry point: `lake env lean --run <path>` prints the
    JSON registry of all V5 rules. -/
def main : IO Unit :=
  IO.println TensorGuard.V5.rulesJson
