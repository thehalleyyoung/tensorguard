# Worked Z3 + refinement-type example: rb_003 (GPT-NeoX QKV view)

## (i) The bug

Real fix-PR `huggingface/transformers#23081`.  The QKV linear projection emits `3 * hidden_size = 3072` features per token, but the subsequent `view` to per-head `(B, T, N, 3*hd)` uses the Python-truncated head size `hd = H // N = 1024 // 12 = 85`, so the view target packs only `N * 3 * hd = 3060` per token.  The shape mismatch is a per-head divisibility violation: `N` does not divide `H`.

## (ii) Refinement-type derivation (rule-driven, no AST pattern)

### `module_config_symbols`
- **H**: `1024`
- **N**: `12`
- **head_size_floor**: `85`
- **B**: `1`
- **T**: `5`

### `T-Input  [x]`
- **rule**: `param-shape`
- **shape**: `[1, 5, 1024]`
- **type**: `Tensor{shape = (B, T, H)}`

### `T-Linear [qkv = self.qkv(x)]`
- **rule**: `T-Linear : in_features = H ✓; out_features = 3*H replaces last dim`
- **in_features**: `1024`
- **out_features**: `3072`
- **shape**: `[1, 5, 3072]`
- **type**: `Tensor{shape = (B, T, 3*H)}`

### `T-View   [qkv.view(1, 5, 12, 255)]`
- **rule**: `T-View : prod(input_shape) == prod(target_shape)`
- **input_shape**: `[1, 5, 3072]`
- **target_shape**: `[1, 5, 12, 255]`
- **obligation**: `B*T*(3*H) == B*T*N*(3*(H//N))   ⇔   N | H  (per-head divisibility)`
- **input_numel**: `15360`
- **target_numel**: `15300`

## (iii) The actual Z3 query

### Concrete instance (rb_003 verbatim shapes)

`validate_reshape_with_z3((1,5,3072), (1,5,12,255))` returns `[False, 'Reshape incompatible: input and target element counts cannot be equal']` and the underlying SMT-LIB2 query (verdict: **unsat**) is:

```smt2
; benchmark generated from python API
(set-info :status unknown)
(assert
 (let ((?x19 (* (* (* (* 1 1) 5) 12) 255)))
(let ((?x13 (* (* 1 1) 5)))
(let ((?x15 (* ?x13 3072)))
(= ?x15 ?x19)))))
(check-sat)
```

### Symbolic instance (module-config symbols)

With `H, N, hd, B, T` as integer symbols and the floor-div semantics `N*hd <= H < N*(hd+1)`, asserting the buggy guard `H=1024, N=12, B=1, T=5` together with the negated T-View obligation `in_numel != tgt_numel` gives Z3 verdict **sat** with witness `{'in_numel': 15360, 'H': 1024, 'N': 12, 'B': 1, 'T': 5, 'tgt_numel': 15300, 'head_size': 85}`.  Under the corrected guard `H=1024, N=16` the same query returns `unsat` (no counterexample exists, i.e.\ the obligation holds).  The verbatim SMT-LIB2 query for the buggy-guard instance:

```smt2
; benchmark generated from python API
(set-info :status unknown)
(declare-fun B () Int)
(declare-fun T () Int)
(declare-fun H () Int)
(declare-fun N () Int)
(declare-fun head_size () Int)
(declare-fun in_numel () Int)
(declare-fun tgt_numel () Int)
(assert
 (>= B 1))
(assert
 (>= T 1))
(assert
 (>= H 1))
(assert
 (>= N 1))
(assert
 (>= head_size 0))
(assert
 (let ((?x43 (* N head_size)))
 (<= ?x43 H)))
(assert
 (< H (* N (+ head_size 1))))
(assert
 (= in_numel (* (* (* B T) 3) H)))
(assert
 (= tgt_numel (* (* (* (* B T) N) 3) head_size)))
(assert
 (= H 1024))
(assert
 (= N 12))
(assert
 (= B 1))
(assert
 (= T 5))
(assert
 (and (distinct in_numel tgt_numel) true))
(check-sat)
```

## (iv) The Refuted-Proof witness

Buggy guard: `H = 1024, N = 12, hd = 85, B = 1, T = 5`.  Z3 satisfies the negation of the T-View obligation with the assignment `in_numel = 15360` (= `B*T*3*H`) and `tgt_numel = 15300` (= `B*T*N*3*hd`); the failed shape predicate is `in_numel == tgt_numel`.  This concrete assignment is the Refuted-Proof witness for rb_003.  Under the corrected guard `N = 16` the same predicate becomes satisfiable (`hd = 64`, both numels = 15360), so Z3 reports `unsat` --- no witness exists --- and the verifier passes.

## (v) Why a pure AST pattern cannot produce this witness

An AST-only matcher can spot a `view(...)` call but cannot decide whether the resulting per-head divisibility obligation `N | H` holds without arithmetic reasoning over module-config symbols.  rb_003 has H=1024, N=12, head_size = H // N = 85 (Python integer division silently drops the remainder).  The shape mismatch (3072 vs 3060 per-token) is a function of an integer-divisibility predicate over the module's constructor symbols; deciding it requires an SMT-style calculus that can multiply, divide and compare integer expressions, which is exactly what the rule-driven Z3 path above does.
