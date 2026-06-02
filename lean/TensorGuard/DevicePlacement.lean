/-
TensorGuard device-placement transfer semantics, machine-checked in Lean 4
(Step 140).

`DeviceDtype.lean` models a single device tag and the *binary-op same-device*
check.  This file lifts it to the **device-placement transfer function over a
chain of `.to(...)` moves**, exactly as the verifier propagates a tensor's
device tag through a `forward`:

  * a device-agnostic op (`keep`, e.g. `relu`, `x * 2`) propagates the incoming
    device tag unchanged;
  * `toCpu` (`.cpu()` / `.to('cpu')`) forces the tag to `cpu`;
  * `toAccel` (`.cuda()` / `.to('mps')` / `.to(accel)`) forces it to the
    accelerator tag.

`devRun d0 ops` folds the per-op transfer over a chain.  A binary op (`add`,
`mul`, `matmul`, ...) between two tensors is *valid* iff both operands carry the
same device tag — exactly the predicate the verifier discharges, and exactly the
condition real torch enforces ("Expected all tensors to be on the same device").

We prove the algebraic laws the verifier relies on: `keep` is the identity, a
move is absorbing (last move wins regardless of the prefix), the run decomposes
over concatenation, the outgoing tag equals the last explicit move (or the input
tag when no move intervened), and the binary-op validity predicate is exactly
tag-equality (so a cross-device pair is *always* flagged — refutation
soundness — and a same-device pair is *never* falsely flagged).

The companion test `tests/test_device_placement_transfer.py` replays each chain
on a **real tensor across `cpu` and `mps`** and asserts `tensor.device.type`
equals `devRun`, plus that a cross-device binary op raises in eager torch exactly
when the Lean `binValid` predicate is `false`.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace DevicePlacement

/-- A device tag.  Two tags suffice to model the same-device check: the host
    (`cpu`) and a single accelerator (`accel`, standing for cuda/mps/...). -/
inductive Dev
  | cpu
  | accel
  deriving DecidableEq, Repr

/-- One step of the device-placement transfer. -/
inductive DevOp
  | keep     -- device-agnostic op: propagate the incoming tag
  | toCpu    -- `.cpu()` / `.to('cpu')`
  | toAccel  -- `.cuda()` / `.to('mps')` / `.to(accel)`
  deriving DecidableEq, Repr

/-- Per-op transfer on the device tag. -/
def devStep (d : Dev) : DevOp → Dev
  | DevOp.keep    => d
  | DevOp.toCpu   => Dev.cpu
  | DevOp.toAccel => Dev.accel

/-- Whether an op *moves* the tensor (resets the tag irrespective of its input). -/
def isMove : DevOp → Bool
  | DevOp.keep => false
  | _ => true

/-- The target of a move op (only meaningful when `isMove` is true). -/
def moveTarget : DevOp → Dev
  | DevOp.toCpu   => Dev.cpu
  | DevOp.toAccel => Dev.accel
  | DevOp.keep    => Dev.cpu  -- unused

/-- Fold the transfer over a chain of ops. -/
def devRun (d0 : Dev) : List DevOp → Dev
  | [] => d0
  | op :: rest => devRun (devStep d0 op) rest

/-- A binary op is valid iff both operands carry the same device tag. -/
def binValid (a b : Dev) : Bool := a == b

/- ===================================================================== -/
/- 1. Per-op laws                                                        -/
/- ===================================================================== -/

theorem keep_id (d : Dev) : devStep d DevOp.keep = d := rfl
theorem toCpu_cpu (d : Dev) : devStep d DevOp.toCpu = Dev.cpu := rfl
theorem toAccel_accel (d : Dev) : devStep d DevOp.toAccel = Dev.accel := rfl

/-- A move op is **absorbing**: its output tag does not depend on the input
    (this is what makes `.to(device)` last-wins). -/
theorem move_absorbing (op : DevOp) (h : isMove op = true) (d d' : Dev) :
    devStep d op = devStep d' op := by
  cases op <;> simp_all [isMove, devStep]

/-- A move op outputs exactly its declared target, independent of input. -/
theorem move_hits_target (op : DevOp) (h : isMove op = true) (d : Dev) :
    devStep d op = moveTarget op := by
  cases op <;> simp_all [isMove, devStep, moveTarget]

/- ===================================================================== -/
/- 2. Chain laws                                                         -/
/- ===================================================================== -/

/-- The run decomposes over concatenation (compositionality of the transfer). -/
theorem devRun_append (d0 : Dev) (xs ys : List DevOp) :
    devRun d0 (xs ++ ys) = devRun (devRun d0 xs) ys := by
  induction xs generalizing d0 with
  | nil => rfl
  | cons op rest ih => simp [devRun, ih]

/-- **Last move wins**: appending a move op makes the whole prefix irrelevant —
    the result is exactly that op's target device, independent of `d0`. -/
theorem run_after_move (d0 d0' : Dev) (xs : List DevOp)
    (op : DevOp) (h : isMove op = true) :
    devRun d0 (xs ++ [op]) = devRun d0' (xs ++ [op]) := by
  rw [devRun_append, devRun_append]
  have : devStep (devRun d0 xs) op = devStep (devRun d0' xs) op :=
    move_absorbing op h _ _
  simp [devRun, this]

/-- The result of a chain ending in a move equals that move's target. -/
theorem run_ends_at_target (d0 : Dev) (xs : List DevOp)
    (op : DevOp) (h : isMove op = true) :
    devRun d0 (xs ++ [op]) = moveTarget op := by
  rw [devRun_append]
  simpa [devRun] using move_hits_target op h (devRun d0 xs)

/-- **Characterization for the move-free fragment**: with no move op the
    outgoing tag is exactly the input tag (device is preserved). -/
theorem run_noMove_id (d0 : Dev) (xs : List DevOp)
    (h : ∀ op ∈ xs, isMove op = false) :
    devRun d0 xs = d0 := by
  induction xs generalizing d0 with
  | nil => rfl
  | cons op rest ih =>
    have hop : isMove op = false := h op (List.mem_cons_self _ _)
    have hrest : ∀ o ∈ rest, isMove o = false :=
      fun o ho => h o (List.mem_cons_of_mem _ ho)
    cases op
    · simp only [devRun, devStep]; exact ih d0 hrest
    · simp [isMove] at hop
    · simp [isMove] at hop

/- ===================================================================== -/
/- 3. Binary-op same-device check                                        -/
/- ===================================================================== -/

/-- Validity is reflexive: a binary op on two same-device operands is valid
    (no false alarm — refutation soundness for the device domain). -/
theorem binValid_refl (d : Dev) : binValid d d = true := by
  simp [binValid]

/-- Validity is exactly tag-equality: a cross-device pair is *always* flagged. -/
theorem binValid_iff_eq (a b : Dev) : binValid a b = true ↔ a = b := by
  simp [binValid]

/-- A concrete cross-device pair is flagged invalid. -/
theorem cpu_accel_invalid : binValid Dev.cpu Dev.accel = false := by
  decide

/-- Whatever the two chains compute, the binary op of their results is valid iff
    they land on the same device — connecting the transfer to the check. -/
theorem chain_binValid_iff (d0 d1 : Dev) (xs ys : List DevOp) :
    binValid (devRun d0 xs) (devRun d1 ys) = true ↔ devRun d0 xs = devRun d1 ys := by
  exact binValid_iff_eq _ _

end DevicePlacement
end TensorGuard
