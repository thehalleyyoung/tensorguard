/-
TensorGuard distribution batch/event/log-prob shape rules, machine-checked in
Lean 4 (Step 234).

`src/distributions_verify.py` models the usable shape contract for selected
`torch.distributions`: constructor batch/event shapes plus the output shape of
`log_prob`.  This file mechanizes the concrete, shape-only core for Normal,
Categorical, MultivariateNormal, Independent, and the identity/reshape fragment
of TransformedDistribution.  Symbolic dimensions and value-support constraints
remain in the Python checker and live PyTorch oracle tests.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace Distributions

/- ===================================================================== -/
/- 1. Shared concrete shape utilities                                    -/
/- ===================================================================== -/

structure DistShape where
  batch : List Nat
  event : List Nat
deriving DecidableEq, Repr

def prod : List Nat → Nat
  | [] => 1
  | x :: xs => x * prod xs

def dropLast : List Nat → List Nat
  | [] => []
  | [_] => []
  | x :: xs => x :: dropLast xs

def last? : List Nat → Option Nat
  | [] => none
  | [x] => some x
  | _ :: xs => last? xs

def trailing2? : List Nat → Option (List Nat × Nat × Nat)
  | [] => none
  | [_] => none
  | [x, y] => some ([], x, y)
  | x :: xs =>
      match trailing2? xs with
      | some (pre, a, b) => some (x :: pre, a, b)
      | none => none

def bcDim? (a b : Nat) : Option Nat :=
  if a = b then
    some a
  else if a = 1 then
    some b
  else if b = 1 then
    some a
  else
    none

def broadcastRev? : List Nat → List Nat → Option (List Nat)
  | [], ys => some ys
  | xs, [] => some xs
  | x :: xs, y :: ys =>
      match bcDim? x y, broadcastRev? xs ys with
      | some z, some zs => some (z :: zs)
      | _, _ => none

def broadcastShapes? (a b : List Nat) : Option (List Nat) :=
  match broadcastRev? a.reverse b.reverse with
  | some out => some out.reverse
  | none => none

theorem bcDim_same (n : Nat) : bcDim? n n = some n := by
  simp [bcDim?]

theorem bcDim_one_left (n : Nat) : bcDim? 1 n = some n := by
  by_cases h : 1 = n <;> simp [bcDim?, h]

theorem bcDim_one_right (n : Nat) : bcDim? n 1 = some n := by
  by_cases h : n = 1 <;> simp [bcDim?, h]

theorem bcDim_incompatible_example : bcDim? 2 3 = none := by
  rfl

theorem broadcast_example :
    broadcastShapes? [2, 1, 3] [1, 4, 3] = some [2, 4, 3] := by
  rfl

theorem broadcast_incompatible_example :
    broadcastShapes? [2] [3] = none := by
  rfl

/- ===================================================================== -/
/- 2. Constructor batch/event rules                                      -/
/- ===================================================================== -/

def normal? (loc scale : List Nat) : Option DistShape :=
  match broadcastShapes? loc scale with
  | some b => some { batch := b, event := [] }
  | none => none

def categorical? (param : List Nat) : Option DistShape :=
  match last? param with
  | some k =>
      if 0 < k then
        some { batch := dropLast param, event := [] }
      else
        none
  | none => none

def multivariateNormal? (loc matrix : List Nat) : Option DistShape :=
  match last? loc, trailing2? matrix with
  | some ev, some (mbatch, left, right) =>
      if left = right then
        if ev = left then
          match broadcastShapes? (dropLast loc) mbatch with
          | some b => some { batch := b, event := [ev] }
          | none => none
        else
          none
      else
        none
  | _, _ => none

def independent? (base : DistShape) (k : Nat) : Option DistShape :=
  if k ≤ base.batch.length then
    let cut := base.batch.length - k
    some {
      batch := base.batch.take cut,
      event := base.batch.drop cut ++ base.event
    }
  else
    none

theorem normal_broadcast_output :
    normal? [2, 1, 3] [1, 4, 3] =
      some { batch := [2, 4, 3], event := [] } := by
  rfl

theorem normal_bad_broadcast_rejected :
    normal? [2] [3] = none := by
  rfl

theorem categorical_batch_drops_category_dim :
    categorical? [2, 3, 4] = some { batch := [2, 3], event := [] } := by
  rfl

theorem categorical_empty_rank_rejected :
    categorical? [] = none := by
  rfl

theorem categorical_zero_categories_rejected :
    categorical? [2, 0] = none := by
  rfl

theorem mvn_batch_event_output :
    multivariateNormal? [4, 3, 5] [1, 3, 5, 5] =
      some { batch := [4, 3], event := [5] } := by
  rfl

theorem mvn_matrix_square_rejected :
    multivariateNormal? [3] [3, 4] = none := by
  rfl

theorem mvn_event_mismatch_rejected :
    multivariateNormal? [3] [4, 4] = none := by
  rfl

theorem mvn_batch_broadcast_rejected :
    multivariateNormal? [2, 3] [4, 3, 3] = none := by
  rfl

theorem independent_moves_batch_to_event :
    independent? { batch := [2, 3, 4], event := [] } 2 =
      some { batch := [2], event := [3, 4] } := by
  rfl

theorem independent_preserves_when_zero :
    independent? { batch := [2, 3], event := [4] } 0 =
      some { batch := [2, 3], event := [4] } := by
  rfl

theorem independent_too_many_rejected :
    independent? { batch := [2, 3], event := [] } 3 = none := by
  rfl

/- ===================================================================== -/
/- 3. log_prob output-shape rule                                         -/
/- ===================================================================== -/

def logProb? (dist : DistShape) (value : List Nat) : Option (List Nat) :=
  match broadcastShapes? value (dist.batch ++ dist.event) with
  | some b =>
      let eventRank := dist.event.length
      some (b.take (b.length - eventRank))
  | none => none

theorem normal_logProb_broadcasts_value :
    logProb? { batch := [2, 3], event := [] } [5, 2, 3] =
      some [5, 2, 3] := by
  rfl

theorem categorical_logProb_drops_no_event :
    logProb? { batch := [2, 3], event := [] } [3] =
      some [2, 3] := by
  rfl

theorem mvn_logProb_drops_event_dim :
    logProb? { batch := [4, 3], event := [5] } [7, 4, 3, 1] =
      some [7, 4, 3] := by
  rfl

theorem logProb_bad_value_broadcast_rejected :
    logProb? { batch := [4, 3], event := [5] } [4, 3, 6] = none := by
  rfl

/- ===================================================================== -/
/- 4. TransformedDistribution identity/reshape fragment                  -/
/- ===================================================================== -/

inductive TransformKind where
  | identity
  | reshape (inputEvent outputEvent : List Nat)
deriving DecidableEq, Repr

open TransformKind

def transformDomainEventDim : TransformKind → Nat
  | identity => 0
  | reshape inputEvent _ => inputEvent.length

def transformCodomainEventDim : TransformKind → Nat
  | identity => 0
  | reshape _ outputEvent => outputEvent.length

def hasSuffix (shape suffix : List Nat) : Bool :=
  decide (suffix.length ≤ shape.length) &&
    decide (shape.drop (shape.length - suffix.length) = suffix)

def reshapeShape? (shape inputEvent outputEvent : List Nat) : Option (List Nat) :=
  if prod inputEvent = prod outputEvent then
    if hasSuffix shape inputEvent then
      some (shape.take (shape.length - inputEvent.length) ++ outputEvent)
    else
      none
  else
    none

def forwardShape? : TransformKind → List Nat → Option (List Nat)
  | identity, shape => some shape
  | reshape inputEvent outputEvent, shape => reshapeShape? shape inputEvent outputEvent

def inverseShape? : TransformKind → List Nat → Option (List Nat)
  | identity, shape => some shape
  | reshape inputEvent outputEvent, shape => reshapeShape? shape outputEvent inputEvent

def forwardRun? : List TransformKind → List Nat → Option (List Nat)
  | [], shape => some shape
  | t :: ts, shape =>
      match forwardShape? t shape with
      | some next => forwardRun? ts next
      | none => none

def inverseRun? : List TransformKind → List Nat → Option (List Nat)
  | [], shape => some shape
  | t :: ts, shape =>
      match inverseRun? ts shape with
      | some prev => inverseShape? t prev
      | none => none

def adjustEventDim (current plus minus : Nat) : Nat :=
  if minus ≤ plus then
    current + (plus - minus)
  else
    current - (minus - plus)

def domainFold : List TransformKind → Nat → Nat
  | [], eventDim => eventDim
  | t :: ts, eventDim =>
      let next := adjustEventDim eventDim (transformDomainEventDim t) (transformCodomainEventDim t)
      domainFold ts (max next (transformDomainEventDim t))

def codomainFold : List TransformKind → Nat → Nat
  | [], eventDim => eventDim
  | t :: ts, eventDim =>
      let next := adjustEventDim eventDim (transformCodomainEventDim t) (transformDomainEventDim t)
      codomainFold ts (max next (transformCodomainEventDim t))

def composeDomainEventDim : List TransformKind → Nat
  | [] => 0
  | t :: ts => domainFold (t :: ts).reverse (transformCodomainEventDim ((t :: ts).getLast (by simp)))

def composeCodomainEventDim : List TransformKind → Nat
  | [] => 0
  | t :: ts => codomainFold (t :: ts) (transformDomainEventDim t)

def transformed? (base : DistShape) (transforms : List TransformKind) : Option DistShape :=
  match transforms with
  | [] => none
  | _ =>
      let baseShape := base.batch ++ base.event
      let domainEventDim := composeDomainEventDim transforms
      let codomainEventDim := composeCodomainEventDim transforms
      if domainEventDim ≤ baseShape.length then
        match forwardRun? transforms baseShape with
        | some forwardShape =>
            match inverseRun? transforms forwardShape with
            | some inverseShape =>
                if inverseShape = baseShape then
                  let shiftedBaseEventDim :=
                    adjustEventDim base.event.length codomainEventDim domainEventDim
                  let eventRank := max codomainEventDim shiftedBaseEventDim
                  if eventRank ≤ forwardShape.length then
                    some {
                      batch := forwardShape.take (forwardShape.length - eventRank),
                      event := forwardShape.drop (forwardShape.length - eventRank)
                    }
                  else
                    none
                else
                  none
            | none => none
        | none => none
      else
        none

theorem reshapeShape_output :
    reshapeShape? [4, 2, 3] [2, 3] [6] = some [4, 6] := by
  rfl

theorem reshapeShape_wrong_suffix_rejected :
    reshapeShape? [4, 2, 3] [3, 3] [9] = none := by
  rfl

theorem reshapeShape_numel_mismatch_rejected :
    reshapeShape? [4, 2, 3] [2, 3] [5] = none := by
  rfl

theorem transformed_identity_preserves_shape :
    transformed? { batch := [2, 3], event := [] } [identity] =
      some { batch := [2, 3], event := [] } := by
  rfl

theorem transformed_reshape_event_shape :
    transformed? { batch := [4], event := [2, 3] } [reshape [2, 3] [6]] =
      some { batch := [4], event := [6] } := by
  rfl

theorem transformed_reshape_reinterprets_batch :
    transformed? { batch := [2, 3], event := [] } [reshape [3] [3]] =
      some { batch := [2], event := [3] } := by
  rfl

theorem transformed_composed_reshape_identity :
    transformed? { batch := [4], event := [2, 3] } [reshape [2, 3] [6], identity] =
      some { batch := [4], event := [6] } := by
  rfl

theorem transformed_wrong_domain_rejected :
    transformed? { batch := [2], event := [] } [reshape [2, 3] [6]] = none := by
  rfl

end Distributions
end TensorGuard
