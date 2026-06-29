# TensorGuard symbolic-execution — Operational Semantics

> Generated from `src/symexec/semantics.py` — the single source of truth. Do not edit by hand; run `python -m src.symexec.semantics > docs/symexec/semantics.md` and it is pinned by `tests/test_symexec_semantics.py`.

## Configurations

A configuration is ⟨c, σ⟩ where c is the residual code (a list of statements with a program counter) and σ is the abstract store.  A method/function body is a sequence of statements executed left-to-right; the engine analyses the `__main__` harness, free top-level functions, and class `forward`/`__call__` methods (bare module-level statements are NOT executed).

## The abstract store σ

σ = (env, store, reachable, dim_facts):

| Component | Meaning | Code |
| --- | --- | --- |
| `env : Name ⇀ AbstractValue` | the variable environment; a missing name reads as ⊤ (unknown). | `src/symexec/state.py:State.env, State.get/set` |
| `store : Obj × Attr ⇀ AbstractValue` | the object/attribute store; `self` is the canonical key for the module instance under analysis, so `self.fc = nn.Linear(...)` set in `__init__` is visible to `forward`. | `src/symexec/state.py:State.store, get_attr/set_attr` |
| `reachable : Bool` | the path reachability flag; an unreachable configuration takes no step and emits no bug. | `src/symexec/state.py:State.reachable` |
| `dim_facts : Constraint*` | path constraints over symbolic dimensions accumulated by guard refinement along the current path; a report whose failing condition is provably UNSAT under dim_facts is suppressed (feasibility gate). | `src/symexec/state.py:State.dim_facts; src/symexec/smt_bridge.py` |

Stores form a lattice: `⊥ ⊑ … ⊑ ⊤`, with a pointwise join `⊔` used at control-flow merges and a widening `▽` used to terminate loops.

## Statements — small-step rules

| Form | `ast` node | Small-step rule (over the abstract store σ) | Code |
| --- | --- | --- | --- |
| `x = e   /   x, y = e   /   *xs = e` | `ast.Assign` | ⟨x = e, σ⟩ ⇒ ⟨skip, σ[x ↦ E⟦e⟧σ]⟩.  Tuple/list targets destructure a TupleVal/ListVal componentwise; an arity mismatch is reported (UNPACK_ARITY_MISMATCH) and the targets bind ⊤. | `src/symexec/interpreter.py:_exec (ast.Assign), _bind_target` |
| `x: T = e   /   x: T` | `ast.AnnAssign` | Same transfer as Assign when a value is present; a bare annotation is a no-op on σ. | `src/symexec/interpreter.py (ast.AnnAssign)` |
| `x op= e` | `ast.AugAssign` | ⟨x op= e, σ⟩ ⇒ ⟨x = x op e, σ⟩: desugars to the binary transfer on the current value of x. | `src/symexec/interpreter.py (ast.AugAssign)` |
| `e   (expression statement)` | `ast.Expr` | ⟨e, σ⟩ ⇒ ⟨skip, σ'⟩ where E⟦e⟧ is evaluated for its bug-checking side effects (e.g. a bare `a @ b`) and σ' carries any attribute writes; the value is discarded. | `src/symexec/interpreter.py (ast.Expr)` |
| `pass` | `ast.Pass` | ⟨pass, σ⟩ ⇒ ⟨skip, σ⟩. | `src/symexec/interpreter.py (ast.Pass)` |
| `return e` | `ast.Return` | Evaluates E⟦e⟧σ, records it as the function's return value (checked against the declared/contracted arity, RETURN_ARITY_CONTRACT), and marks the continuation unreachable. | `src/symexec/interpreter.py (ast.Return)` |
| `if e: S1 else: S2` | `ast.If` | Evaluates the guard; when its truth value is unknown the engine explores BOTH branches from refined copies of σ (σ ⊓ e, σ ⊓ ¬e — guard refinement adds to dim_facts) and JOINS the resulting stores at the merge point: ⟨if e …, σ⟩ ⇒ ⟨skip, σ1 ⊔ σ2⟩.  A statically decidable guard prunes the dead branch (reachable=False). | `src/symexec/interpreter.py (ast.If); State.join; src/symexec/relational.py` |
| `for x in e: S` | `ast.For` | Bounded unrolling toward a fixpoint: the loop body is re-executed on the joined store until σ stabilises, applying WIDENING after a bounded number of iterations to guarantee termination; the iterate value binds the element abstraction of E⟦e⟧σ (⊤ if unknown). | `src/symexec/interpreter.py (ast.For); State.widen; ITERATION_CAPS` |
| `while e: S` | `ast.While` | Same fixpoint/widening treatment as `for`, iterating on the guard-refined store until a post-fixpoint is reached or the iteration cap triggers widening to ⊤ on the unstable slots. | `src/symexec/interpreter.py (ast.While); State.widen; ITERATION_CAPS` |
| `with e as x: S` | `ast.With` | Binds x to E⟦e⟧σ (the context value, ⊤ if unmodeled) and executes the body; no special enter/exit effect is modeled. | `src/symexec/interpreter.py (ast.With)` |
| `assert e` | `ast.Assert` | Refines σ by the asserted condition (adds to dim_facts) on the continuation; never used to emit a bug. | `src/symexec/interpreter.py (ast.Assert)` |
| `def f(...): S   (nested)` | `ast.FunctionDef` | A def binds a callable summary in σ; calls are handled by E⟦·⟧ via inlining/summary application (interprocedural). | `src/symexec/interpreter.py (ast.FunctionDef); _analyze_function` |

## Expressions — evaluation `E⟦e⟧σ`

| Form | `ast` node | Small-step rule (over the abstract store σ) | Code |
| --- | --- | --- | --- |
| `literal / None` | `ast.Constant` | E⟦c⟧σ = the corresponding IntVal/FloatVal/StrVal/BoolVal/NoneVal. | `src/symexec/interpreter.py (ast.Constant); values.py` |
| `x` | `ast.Name` | E⟦x⟧σ = σ.env(x), or ⊤ if unbound. | `src/symexec/interpreter.py (ast.Name)` |
| `obj.attr` | `ast.Attribute` | E⟦self.a⟧σ = σ.store(self, a); `.shape`/`.dtype`/`.T`/`.ndim` etc. are modeled projections of a TensorVal. | `src/symexec/interpreter.py (ast.Attribute)` |
| `(e1, …)` | `ast.Tuple` | Construct a TupleVal of the evaluated elements; a `*e` element splices a sequence value. | `src/symexec/interpreter.py (ast.Tuple/Starred)` |
| `[e1, …]` | `ast.List` | Construct a ListVal of the evaluated elements. | `src/symexec/interpreter.py (ast.List)` |
| `{k: v, …}` | `ast.Dict` | Construct a DictVal of the evaluated key/value pairs. | `src/symexec/interpreter.py (ast.Dict)` |
| `e1 op e2  (incl. @)` | `ast.BinOp` | Applies the operator transfer function: arithmetic on IntVal/FloatVal (with DIVISION_BY_ZERO when the divisor is provably 0), and `@` invokes the matmul shape transfer (MATMUL_DIM_MISMATCH) propagating the result shape. | `src/symexec/interpreter.py (ast.BinOp); transfer.py` |
| `op e  (-e, not e, ~e)` | `ast.UnaryOp` | Unary transfer on the operand abstraction (e.g. USub on a known IntVal). | `src/symexec/interpreter.py (ast.UnaryOp)` |
| `e1 and e2  /  e1 or e2` | `ast.BoolOp` | Short-circuit boolean abstraction over operand truth values; result is BoolVal or ⊤. | `src/symexec/interpreter.py (ast.BoolOp)` |
| `e1 < e2  (and other comparisons)` | `ast.Compare` | Yields a BoolVal when both sides are concrete/related, else ⊤; feeds guard refinement in `if`/`while`/`assert`. | `src/symexec/interpreter.py (ast.Compare)` |
| `e1 if c else e2` | `ast.IfExp` | Joins the two branch abstractions: E⟦e1 if c else e2⟧σ = E⟦e1⟧σ ⊔ E⟦e2⟧σ (refined by c when decidable). | `src/symexec/interpreter.py (ast.IfExp)` |
| `e[i]  /  e[a:b]` | `ast.Subscript` | Index/slice transfer: integer indexing checks bounds against a known length (TENSOR_INDEX_OOB / RANK_INDEX_ERROR) and projects the element/subshape; symbolic indices abstain. | `src/symexec/interpreter.py (ast.Subscript); _eval_subscript` |
| `f(e1, …, *args, **kw)` | `ast.Call` | The interprocedural / library transfer: torch ops, `nn` constructors and their `forward` application, einops/einsum, builtins (`len`, `range`, `map`/`filter`), user functions (inlined), and stubbed library calls all dispatch here, each with its own sound shape transfer + bug check. | `src/symexec/interpreter.py:_eval_call; stubs.py; transfer.py` |
| `[e for x in it if c]` | `ast.ListComp` | Models the comprehension as a bounded loop producing a ListVal of the element abstraction; unbounded/opaque iterables yield a ListVal of ⊤. | `src/symexec/interpreter.py (ast.ListComp)` |

## Abstraction & soundness notes

* Out-of-fragment forms (data-dependent control flow that the guard model cannot refine, dynamic eval/exec, `*`-imports of opaque modules, comprehensions over opaque iterables, etc.) are handled by abstracting the affected values to ⊤ and, where a sound result cannot be guaranteed, by ABSTAINING (recording an AbstainReason and emitting no bug) rather than guessing — this is what keeps the reported set free of false positives.
* Control-flow joins use the store join ⊔ (pointwise value join); loops use widening ▽ after a bounded iteration count (ITERATION_CAPS) to force termination at a sound post-fixpoint.
* The abstract relation ⇒ is designed so that for every concrete step ⟨s, σ_c⟩ → ⟨s', σ_c'⟩ there is an abstract step ⟨s, α(σ_c)⟩ ⇒ ⟨s', σ#⟩ with α(σ_c') ⊑ σ# (local soundness of each transfer function); composing these steps gives the whole-program soundness theorem of Step 92.
* Each per-operator transfer function is the abstraction of the operator's concrete shape semantics; Step 93 formalises (α, γ) as a Galois connection so that 'abstraction of the concrete transfer ⊑ abstract transfer' is the single proof obligation per operator.

