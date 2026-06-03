/-
TensorGuard whole-module subject reduction for a straight-line tensor language.

This module composes the per-operator transfer rules into a small whole-module
language: statements read previously-bound tensors from an environment and append
one or more outputs.  The theorem is proof-carrying by design.  Primitive
families (pointwise, linear, flatten/view/reshape, matmul, indexing,
normalization, attention, etc.) have direct constructors; registry operators
whose local shape rule is not primitive in this file enter through
`certifiedExternal`, which requires the local transfer to provide well-formed
output shapes before the whole-program theorem can apply.

Consequently the main theorem proves the compositional property that matters for
the verifier: once every local complete/sound transfer succeeds, executing any
straight-line module preserves tensor well-formedness for every intermediate and
final tensor.  Per-operator correspondence to PyTorch is checked by the Python
conformance tests and by the existing operator-specific Lean modules.
-/

namespace TensorGuard
namespace SubjectReduction

abbrev Sh := List Nat
abbrev Env := List Sh

/-- Tensor shapes in this straight-line fragment use positive concrete dims.
Zero-size tensor edge cases are outside this theorem and remain a verifier/tested
surface, not part of the positivity-preservation proof. -/
def PosShape (s : Sh) : Prop :=
  ∀ d, d ∈ s → 0 < d

def ShapesWF (xs : List Sh) : Prop :=
  ∀ s, s ∈ xs → PosShape s

def EnvWF (env : Env) : Prop :=
  ShapesWF env

def prod : Sh → Nat
  | [] => 1
  | d :: rest => d * prod rest

theorem pos_append {a b : Sh} (ha : PosShape a) (hb : PosShape b) :
    PosShape (a ++ b) := by
  intro d hd
  cases List.mem_append.mp hd with
  | inl h => exact ha d h
  | inr h => exact hb d h

theorem pos_singleton {d : Nat} (hd : 0 < d) : PosShape [d] := by
  intro x hx
  simp at hx
  rw [hx]
  exact hd

theorem pos_cons {d : Nat} {rest : Sh} (hd : 0 < d) (hr : PosShape rest) :
    PosShape (d :: rest) := by
  intro x hx
  simp at hx
  cases hx with
  | inl h =>
      rw [h]
      exact hd
  | inr h => exact hr x h

theorem prod_pos {s : Sh} (hs : PosShape s) : 0 < prod s := by
  induction s with
  | nil =>
      simp [prod]
  | cons d rest ih =>
      have hd : 0 < d := hs d (by simp)
      have hr : PosShape rest := by
        intro x hx
        exact hs x (by simp [hx])
      exact Nat.mul_pos hd (ih hr)

theorem shapesWF_append {a b : List Sh} (ha : ShapesWF a) (hb : ShapesWF b) :
    ShapesWF (a ++ b) := by
  intro s hs
  cases List.mem_append.mp hs with
  | inl h => exact ha s h
  | inr h => exact hb s h

theorem shapesWF_singleton {s : Sh} (hs : PosShape s) : ShapesWF [s] := by
  intro t ht
  simp at ht
  rw [ht]
  exact hs

theorem shapesWF_pair {a b : Sh} (ha : PosShape a) (hb : PosShape b) :
    ShapesWF [a, b] := by
  intro s hs
  simp at hs
  cases hs with
  | inl h =>
      rw [h]
      exact ha
  | inr h =>
      simp at h
      rw [h]
      exact hb

/-! ## Straight-line operator families -/

inductive SLOp
  | pointwise (name : String) (shape : Sh)
  | linear (pre : Sh) (inF outF : Nat)
  | flatten (pre span suf : Sh)
  | view (input out : Sh)
  | reshape (input out : Sh)
  | unsqueeze (pre suf : Sh)
  | squeeze (pre suf : Sh)
  | matmul (rest : Sh) (m k n : Nat)
  | bmm (b m k n : Nat)
  | broadcastTo (input target : Sh)
  | binaryBroadcast (left right out : Sh)
  | cat2 (pre post : Sh) (a b : Nat)
  | stack2 (shape : Sh)
  | gather (input index : Sh)
  | scatter (input index src : Sh)
  | indexSelect (pre post : Sh) (old len : Nat)
  | narrow (pre post : Sh) (old len : Nat)
  | embedding (input : Sh) (embDim : Nat)
  | layerNorm (pre norm : Sh)
  | sdpa (batch : Sh) (l s e ev : Nat)
  | certifiedExternal (name : String) (inputs outputs : List Sh)
  deriving Repr, DecidableEq

def inputShapes : SLOp → List Sh
  | .pointwise _ shape => [shape]
  | .linear pre inF _ => [pre ++ [inF]]
  | .flatten pre span suf => [pre ++ span ++ suf]
  | .view input _ => [input]
  | .reshape input _ => [input]
  | .unsqueeze pre suf => [pre ++ suf]
  | .squeeze pre suf => [pre ++ [1] ++ suf]
  | .matmul rest m k n => [rest ++ [m, k], rest ++ [k, n]]
  | .bmm b m k n => [[b, m, k], [b, k, n]]
  | .broadcastTo input _ => [input]
  | .binaryBroadcast left right _ => [left, right]
  | .cat2 pre post a b => [pre ++ [a] ++ post, pre ++ [b] ++ post]
  | .stack2 shape => [shape, shape]
  | .gather input index => [input, index]
  | .scatter input index src => [input, index, src]
  | .indexSelect pre post old len => [pre ++ [old] ++ post, [len]]
  | .narrow pre post old _ => [pre ++ [old] ++ post]
  | .embedding input _ => [input]
  | .layerNorm pre norm => [pre ++ norm]
  | .sdpa batch l s e ev => [batch ++ [l, e], batch ++ [s, e], batch ++ [s, ev]]
  | .certifiedExternal _ inputs _ => inputs

def outputShapes : SLOp → List Sh
  | .pointwise _ shape => [shape]
  | .linear pre _ outF => [pre ++ [outF]]
  | .flatten pre span suf => [pre ++ [prod span] ++ suf]
  | .view _ out => [out]
  | .reshape _ out => [out]
  | .unsqueeze pre suf => [pre ++ [1] ++ suf]
  | .squeeze pre suf => [pre ++ suf]
  | .matmul rest m _ n => [rest ++ [m, n]]
  | .bmm b m _ n => [[b, m, n]]
  | .broadcastTo _ target => [target]
  | .binaryBroadcast _ _ out => [out]
  | .cat2 pre post a b => [pre ++ [a + b] ++ post]
  | .stack2 shape => [[2] ++ shape]
  | .gather _ index => [index]
  | .scatter input _ _ => [input]
  | .indexSelect pre post _ len => [pre ++ [len] ++ post]
  | .narrow pre post _ len => [pre ++ [len] ++ post]
  | .embedding input embDim => [input ++ [embDim]]
  | .layerNorm pre norm => [pre ++ norm]
  | .sdpa batch l _ _ ev => [batch ++ [l, ev]]
  | .certifiedExternal _ _ outputs => outputs

/-- Local proof obligation for a transfer rule.  For primitive constructors this
is exactly the static positivity needed by their output rule.  For complex
registry operators it is the proof-carrying certificate supplied by the local
operator verifier. -/
def OpWF (op : SLOp) : Prop :=
  ShapesWF (outputShapes op)

theorem op_outputs_wf (op : SLOp) (h : OpWF op) :
    ShapesWF (outputShapes op) :=
  h

/-! ## Environments and execution -/

def gather (env : Env) : List Nat → Option (List Sh)
  | [] => some []
  | i :: rest =>
      match env.get? i, gather env rest with
      | some s, some ss => some (s :: ss)
      | _, _ => none

structure Stmt where
  op : SLOp
  inputs : List Nat
  deriving Repr, DecidableEq

def WellTypedStmt (env : Env) (stmt : Stmt) : Prop :=
  gather env stmt.inputs = some (inputShapes stmt.op) ∧ OpWF stmt.op

def step (env : Env) (stmt : Stmt) : Option Env :=
  match gather env stmt.inputs with
  | some shapes =>
      if shapes = inputShapes stmt.op then
        some (env ++ outputShapes stmt.op)
      else
        none
  | none => none

theorem step_of_well_typed {env : Env} {stmt : Stmt}
    (h : WellTypedStmt env stmt) :
    step env stmt = some (env ++ outputShapes stmt.op) := by
  rcases h with ⟨hgather, _hop⟩
  simp [step, hgather]

theorem step_subject_reduction {env env' : Env} {stmt : Stmt}
    (hEnv : EnvWF env)
    (hStmt : WellTypedStmt env stmt)
    (hStep : step env stmt = some env') :
    EnvWF env' := by
  have hExact := step_of_well_typed hStmt
  rw [hExact] at hStep
  cases hStep
  exact shapesWF_append hEnv (op_outputs_wf stmt.op hStmt.2)

abbrev Program := List Stmt

def exec : Env → Program → Option Env
  | env, [] => some env
  | env, stmt :: rest =>
      match step env stmt with
      | some env' => exec env' rest
      | none => none

inductive ProgramWT : Env → Program → Prop
  | nil (env : Env) : ProgramWT env []
  | cons {env : Env} {stmt : Stmt} {rest : Program} :
      WellTypedStmt env stmt →
      ProgramWT (env ++ outputShapes stmt.op) rest →
      ProgramWT env (stmt :: rest)

theorem exec_subject_reduction {env env' : Env} {program : Program}
    (hEnv : EnvWF env)
    (hWT : ProgramWT env program)
    (hExec : exec env program = some env') :
    EnvWF env' := by
  induction program generalizing env env' with
  | nil =>
      simp [exec] at hExec
      cases hExec
      exact hEnv
  | cons stmt rest ih =>
      cases hWT with
      | cons hStmt hRest =>
          have hStep := step_of_well_typed hStmt
          simp [exec, hStep] at hExec
          have hEnvNext : EnvWF (env ++ outputShapes stmt.op) :=
            shapesWF_append hEnv (op_outputs_wf stmt.op hStmt.2)
          exact ih hEnvNext hRest hExec

theorem whole_module_subject_reduction {env env' : Env} {program : Program}
    (hEnv : EnvWF env)
    (hWT : ProgramWT env program)
    (hExec : exec env program = some env') :
    EnvWF env' :=
  exec_subject_reduction hEnv hWT hExec

theorem program_outputs_have_positive_shapes {env env' : Env} {program : Program}
    (hEnv : EnvWF env)
    (hWT : ProgramWT env program)
    (hExec : exec env program = some env') :
    ∀ s, s ∈ env' → PosShape s :=
  whole_module_subject_reduction hEnv hWT hExec

/-! ## Current non-heuristic registry surface

The list is intentionally just the current registry surface, not a per-op shape
semantics proof.  The theorem above covers every listed name through either a
primitive constructor or a `certifiedExternal` local transfer.  The Python test
keeps this list synchronized with `operator_confidence_table.json` and rejects
heuristic operators.
-/

def currentCompleteSoundOperatorNames : List String := [
  "F.celu",
  "F.elu",
  "F.gelu",
  "F.hardshrink",
  "F.hardsigmoid",
  "F.hardswish",
  "F.leaky_relu",
  "F.logsigmoid",
  "F.mish",
  "F.prelu",
  "F.relu",
  "F.rrelu",
  "F.selu",
  "F.sigmoid",
  "F.silu",
  "F.softplus",
  "F.softshrink",
  "F.softsign",
  "F.tanh",
  "F.tanhshrink",
  "torch.abs",
  "torch.acos",
  "torch.all",
  "torch.amax",
  "torch.amin",
  "torch.any",
  "torch.argsort",
  "torch.asin",
  "torch.atan",
  "torch.bernoulli",
  "torch.bmm",
  "torch.broadcast_shapes",
  "torch.broadcast_tensors",
  "torch.cdist",
  "torch.ceil",
  "torch.celu",
  "torch.clamp",
  "torch.clip",
  "torch.column_stack",
  "torch.cos",
  "torch.cosh",
  "torch.cross",
  "torch.dstack",
  "torch.elu",
  "torch.eq",
  "torch.equal",
  "torch.erf",
  "torch.erfc",
  "torch.exp",
  "torch.fft.fft",
  "torch.fft.fft2",
  "torch.fft.ifft",
  "torch.fft.ifft2",
  "torch.fft.irfft",
  "torch.fft.rfft",
  "torch.flip",
  "torch.floor",
  "torch.gather",
  "torch.ge",
  "torch.gelu",
  "torch.gt",
  "torch.hardshrink",
  "torch.hardsigmoid",
  "torch.hardswish",
  "torch.hstack",
  "torch.index_select",
  "torch.isfinite",
  "torch.isinf",
  "torch.isnan",
  "torch.kron",
  "torch.le",
  "torch.leaky_relu",
  "torch.linalg.cholesky",
  "torch.linalg.eig",
  "torch.linalg.inv",
  "torch.linalg.qr",
  "torch.linalg.solve",
  "torch.linalg.svd",
  "torch.log",
  "torch.log10",
  "torch.log2",
  "torch.logsigmoid",
  "torch.logsumexp",
  "torch.lt",
  "torch.matmul",
  "torch.max",
  "torch.mean",
  "torch.min",
  "torch.mish",
  "torch.mm",
  "torch.moveaxis",
  "torch.movedim",
  "torch.mv",
  "torch.nan_to_num",
  "torch.ne",
  "torch.neg",
  "torch.norm",
  "torch.outer",
  "torch.poisson",
  "torch.prelu",
  "torch.prod",
  "torch.relu",
  "torch.repeat_interleave",
  "torch.roll",
  "torch.rot90",
  "torch.round",
  "torch.row_stack",
  "torch.rrelu",
  "torch.rsqrt",
  "torch.scatter",
  "torch.selu",
  "torch.sigmoid",
  "torch.sign",
  "torch.silu",
  "torch.sin",
  "torch.sinh",
  "torch.softplus",
  "torch.softshrink",
  "torch.softsign",
  "torch.sort",
  "torch.sqrt",
  "torch.squeeze",
  "torch.stack",
  "torch.std",
  "torch.sum",
  "torch.swapaxes",
  "torch.swapdims",
  "torch.tan",
  "torch.tanh",
  "torch.tanhshrink",
  "torch.tensordot",
  "torch.tile",
  "torch.topk",
  "torch.unsqueeze",
  "torch.var",
  "torch.vstack"
]

/-! ## Concrete theorem-shaped straight-line modules -/

def mlpProgram : Program := [
  ⟨.linear [2] 8 16, [0]⟩,
  ⟨.pointwise "torch.relu" [2, 16], [1]⟩,
  ⟨.layerNorm [2] [16], [2]⟩,
  ⟨.linear [2] 16 4, [3]⟩
]

theorem mlp_exec_shape :
    exec [[2, 8]] mlpProgram =
      some [[2, 8], [2, 16], [2, 16], [2, 16], [2, 4]] := by
  decide

def cnnHeadProgram : Program := [
  ⟨.certifiedExternal "nn.Conv2d" [[2, 3, 8, 8]] [[2, 6, 8, 8]], [0]⟩,
  ⟨.pointwise "torch.relu" [2, 6, 8, 8], [1]⟩,
  ⟨.flatten [2] [6, 8, 8] [], [2]⟩,
  ⟨.linear [2] 384 10, [3]⟩
]

theorem cnn_head_exec_shape :
    exec [[2, 3, 8, 8]] cnnHeadProgram =
      some [[2, 3, 8, 8], [2, 6, 8, 8], [2, 6, 8, 8], [2, 384], [2, 10]] := by
  decide

def indexingProgram : Program := [
  ⟨.gather [4, 3] [4, 2], [0, 1]⟩,
  ⟨.linear [4] 2 7, [2]⟩
]

theorem indexing_exec_shape :
    exec [[4, 3], [4, 2]] indexingProgram =
      some [[4, 3], [4, 2], [4, 2], [4, 7]] := by
  decide

def attentionProgram : Program := [
  ⟨.sdpa [2, 4] 5 7 8 9, [0, 1, 2]⟩,
  ⟨.linear [2, 4, 5] 9 16, [3]⟩
]

theorem attention_exec_shape :
    exec [[2, 4, 5, 8], [2, 4, 7, 8], [2, 4, 7, 9]] attentionProgram =
      some [[2, 4, 5, 8], [2, 4, 7, 8], [2, 4, 7, 9], [2, 4, 5, 9], [2, 4, 5, 16]] := by
  decide

end SubjectReduction
end TensorGuard
