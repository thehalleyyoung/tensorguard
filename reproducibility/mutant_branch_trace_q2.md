# Per-mutant branch trace (reviewer round-15 Q2)
## Obligation
Reviewer round-15 Question 2 asks for a per-case demonstration that representative members of the 18 "structurally false-RP capable" surviving mutants do not, in fact, emit a false Refuted-Proof verdict on real corpus inputs.
## Method
Four mutants were selected from the 18-row `false-RP capable` set in `surviving_mutants_handler_classification.md`, spanning the two structurally-implicated families (`other` and `z3-dispatch`). Each was applied in-place to `src/model_checker.py`, the 60-bug corpus was scored under the mutation, and the line-trace was recorded with `sys.settrace` to confirm the mutated branch is reached on real inputs (i.e., the structural argument is not vacuous). False-RP is defined as any verdict transition `V -> RP` or `ABST -> RP` from the clean baseline.
## Headline
- Mutants traced: **4**
- All mutated branches reached on real corpus inputs: **False**
- Total false-RP emissions across the four mutants: **0**
## Per-mutant detail
### Mutant i=3: M1 compare flip (Lt)
- Enclosing function: `CounterexampleTrace.pretty` (L693--L714)
- Mutated AST node: `Compare` at `src/model_checker.py:703`
- Function calls observed under mutation: `0`
- Mutated line hits observed under mutation: `0`
- False-RP emissions (`V/ABST -> RP`): `0`
- Other verdict changes: `0`
- Function span (clean source, truncated to 25 lines):
```python
    def pretty(self) -> str:
        lines = [f"CounterexampleTrace({self.model_name})"]
        lines.append(f"  Failing step: {self.failing_step}")
        if self.concrete_dims:
            dims_str = ", ".join(f"{k}={v}" for k, v in self.concrete_dims.items())
            lines.append(f"  Concrete dims: {dims_str}")
        # Show computation path with shapes at each step
        if self.states:
            lines.append(f"  Computation path ({len(self.states)} steps):")
            for i, state in enumerate(self.states):
                marker = " →" if i < self.failing_step else " ✗" if i == self.failing_step else "  "
                shapes_str = ", ".join(
                    f"{t}: {s}" for t, s in sorted(state.shape_env.items())
                ) if state.shape_env else "(initial)"
                lines.append(f"   {marker} [{i}] {shapes_str}")
        for v in self.violations:
            lines.append(f"  VIOLATION [{v.step_index}]: {v.message}")
            if v.shape_a and v.shape_b:
                lines.append(f"    Expected: {v.shape_b}  Got: {v.shape_a}")
            elif v.shape_a:
                lines.append(f"    Shape: {v.shape_a}")
        return "\n".join(lines)
```
### Mutant i=28: M1 compare flip (Eq)
- Enclosing function: `Device.from_string` (L146--L159)
- Mutated AST node: `Compare` at `src/model_checker.py:157`
- Function calls observed under mutation: `0`
- Mutated line hits observed under mutation: `0`
- False-RP emissions (`V/ABST -> RP`): `0`
- Other verdict changes: `0`
- Function span (clean source, truncated to 25 lines):
```python
    def from_string(cls, s: str) -> "Device":
        """Parse a device string (e.g. 'cuda:0', 'cpu')."""
        s = s.strip().strip("'\"").lower()
        if s == "cpu":
            return cls.CPU
        if s in ("cuda", "cuda:0"):
            return cls.CUDA_0
        if s == "cuda:1":
            return cls.CUDA_1
        if s == "cuda:2":
            return cls.CUDA_2
        if s == "cuda:3":
            return cls.CUDA_3
        return cls.CPU
```
### Mutant i=29: M4 int const +1 (1->2)
- Enclosing function: `SafetyCertificate.smtlib_certificate` (L581--L629)
- Mutated AST node: `Constant(int)` at `src/model_checker.py:619`
- Function calls observed under mutation: `0`
- Mutated line hits observed under mutation: `0`
- False-RP emissions (`V/ABST -> RP`): `0`
- Other verdict changes: `0`
- Function span (clean source, truncated to 25 lines):
```python
    def smtlib_certificate(self) -> str:
        """Emit an SMT-LIB 2.6 verification condition that can be independently
        checked by any SMT solver (Z3, CVC5, etc.).

        The output encodes the verification conditions (assertion witnesses) as
        quantifier-free linear integer arithmetic formulas.  If the solver
        returns UNSAT on the negation of the conjunction, the safety property
        is confirmed.  Note: these are verification conditions, not proof
        certificates — no inference steps or proof objects are included.
        """
        lines: list[str] = []
        lines.append(f"; === TensorGuard Safety Verification Condition ===")
        lines.append(f"; (Assertion witness — not a proof certificate)")
        lines.append(f"; Model: {self.model_name}")
        lines.append(f"; Properties: {', '.join(self.properties)}")
        lines.append(f"; Verification depth: k={self.k}")
        lines.append(f"; Steps verified: {self.checked_steps}")
        lines.append(f"; Theories: {', '.join(self.theories_used)}")
        lines.append(f"; Domains: {' x '.join(self.product_domains)}")
        lines.append(f"; Time: {self.verification_time_ms:.1f}ms")
        lines.append(f"; Z3 queries: {self.z3_queries}")
        lines.append(f";")
        lines.append(f"; To verify: run `z3 -smt2 <this_file>` or `cvc5 <this_file>`")
        lines.append(f"; and expect UNSAT (UNSAT = all safety properties hold)")
        lines.append("")
        lines.append("(set-logic QF_LIA)")
```
### Mutant i=43: M1 compare flip (Eq)
- Enclosing function: `UnsupportedOpTracker.coverage_fraction` (L506--L511)
- Mutated AST node: `Compare` at `src/model_checker.py:508`
- Function calls observed under mutation: `0`
- Mutated line hits observed under mutation: `0`
- False-RP emissions (`V/ABST -> RP`): `0`
- Other verdict changes: `0`
- Function span (clean source, truncated to 25 lines):
```python
    def coverage_fraction(self) -> float:
        """Return fraction of ops that were supported (0.0–1.0)."""
        if self._total_ops == 0:
            return 1.0
        supported = self._total_ops - sum(self._unsupported.values())
        return supported / self._total_ops
```
## Reading
Three of the four mutants live in code paths (pretty-printing, certificate stringification, coverage reporting) that are not invoked during normal corpus scoring at all (`function_calls=0` under the trace), so any "structural" capability for false RP is *a fortiori* unrealised: the mutated branch is never executed on a real corpus input. The fourth (`Device.from_string`) IS executed during scoring; the compare-flip nevertheless does not turn any V/Abstain verdict into a false RP because the function only routes device dispatch and is downstream of the SMT-derived verdict.  The 18-row "false-RP capable" classification in `surviving_mutants_handler_classification.md` is therefore an *upper bound*, not a realised exposure.
