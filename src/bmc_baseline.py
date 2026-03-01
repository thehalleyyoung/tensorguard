"""
Bounded Model Checking (BMC) Baseline for nn.Module Verification.

Encodes the entire acyclic computation graph as a single monolithic SMT
query — no CEGAR, no iterative refinement, no guard harvesting.

For acyclic computation graphs, BMC is complete without abstraction:
  - UNSAT ⟹ SAFE  (no counterexample exists)
  - SAT   ⟹ UNSAFE (counterexample found)
  - timeout ⟹ UNKNOWN

This serves as a baseline to evaluate whether CEGAR-style iterative
refinement adds value over monolithic BMC for TensorGuard verification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    ConstraintVerifier,
    CounterexampleTrace,
    Device,
    LayerDef,
    ModelState,
    OpKind,
    Phase,
    SafetyCertificate,
    SafetyViolation,
    VerificationResult,
    extract_computation_graph,
    verify_model,
)
from src.tensor_shapes import TensorShape, ShapeDim

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


class BMCVerdict(Enum):
    """Outcome of a monolithic BMC verification run."""
    SAFE = auto()
    UNSAFE = auto()
    UNKNOWN = auto()


@dataclass
class BMCResult:
    """Result of a monolithic BMC verification run.

    Attributes
    ----------
    verdict : BMCVerdict
        SAFE if UNSAT (model verified), UNSAFE if SAT (bug found),
        UNKNOWN on timeout or solver failure.
    verification_result : VerificationResult or None
        The underlying ConstraintVerifier result (reused encoding).
    time_ms : float
        Wall-clock time for the entire BMC run in milliseconds.
    num_constraints : int
        Total number of Z3 constraints in the monolithic query.
    num_steps : int
        Number of computation steps in the graph.
    counterexample : CounterexampleTrace or None
        Concrete counterexample if verdict is UNSAFE.
    z3_queries : int
        Number of Z3 check() calls made.
    """
    verdict: BMCVerdict = BMCVerdict.UNKNOWN
    verification_result: Optional[VerificationResult] = None
    time_ms: float = 0.0
    num_constraints: int = 0
    num_steps: int = 0
    counterexample: Optional[CounterexampleTrace] = None
    z3_queries: int = 0

    @property
    def safe(self) -> bool:
        return self.verdict == BMCVerdict.SAFE

    def summary(self) -> str:
        return (
            f"BMC: {self.verdict.name}, {self.num_steps} steps, "
            f"{self.num_constraints} constraints, {self.time_ms:.1f}ms"
        )


def verify_model_bmc(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    timeout: int = 60,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    constraints: Optional[Dict[str, Union[str, int]]] = None,
) -> BMCResult:
    """Monolithic BMC verification of an nn.Module.

    Encodes ALL shape + device + phase + gradient constraints into a
    single Z3 query (no CEGAR iteration, no guard harvesting).

    Parameters
    ----------
    source : str
        Python source code containing an ``nn.Module`` subclass.
    input_shapes : dict, optional
        Mapping from forward-parameter names to shape tuples.
    timeout : int
        Z3 solver timeout in seconds (default 60).
    default_device : Device
        Default device for input tensors.
    default_phase : Phase
        Default phase (TRAIN or EVAL).
    constraints : dict, optional
        Relational constraints between symbolic dimensions.

    Returns
    -------
    BMCResult
        Contains verdict (SAFE/UNSAFE/UNKNOWN) and timing information.
    """
    t0 = time.monotonic()

    # --- Extract computation graph ---
    try:
        graph = extract_computation_graph(source)
    except (ValueError, SyntaxError) as exc:
        return BMCResult(
            verdict=BMCVerdict.UNKNOWN,
            time_ms=(time.monotonic() - t0) * 1000,
        )

    if graph.num_steps == 0:
        return BMCResult(
            verdict=BMCVerdict.SAFE,
            time_ms=(time.monotonic() - t0) * 1000,
            num_steps=0,
            num_constraints=0,
        )

    # --- Set Z3 global timeout ---
    if HAS_Z3:
        z3.set_param("timeout", timeout * 1000)

    # --- Build the ConstraintVerifier (reuse existing encoding) ---
    # The ConstraintVerifier already encodes all constraints and walks
    # the full computation graph in _bmc_base_case. We reuse it as the
    # "monolithic" BMC encoder — it already asserts all constraints in
    # a single solver context without any iterative refinement.
    checker = ConstraintVerifier(
        graph,
        input_shapes=input_shapes or {},
        default_device=default_device,
        default_phase=default_phase,
        max_k=graph.num_steps,  # full unrolling
        constraints=constraints,
    )

    # --- Run the monolithic verification (single pass, no CEGAR) ---
    vresult = checker.verify()

    elapsed = (time.monotonic() - t0) * 1000

    # Reset Z3 timeout
    if HAS_Z3:
        z3.set_param("timeout", 0)

    # --- Extract statistics ---
    stats = checker.ctx.get_stats() if HAS_Z3 else {}
    num_constraints = stats.get("z3_queries", 0)
    z3_queries = stats.get("z3_queries", 0)

    # --- Map VerificationResult to BMCVerdict ---
    if vresult.safe:
        verdict = BMCVerdict.SAFE
    elif vresult.counterexample and vresult.counterexample.violations:
        verdict = BMCVerdict.UNSAFE
    elif vresult.errors:
        verdict = BMCVerdict.UNKNOWN
    else:
        verdict = BMCVerdict.UNSAFE

    return BMCResult(
        verdict=verdict,
        verification_result=vresult,
        time_ms=elapsed,
        num_constraints=num_constraints,
        num_steps=graph.num_steps,
        counterexample=vresult.counterexample if not vresult.safe else None,
        z3_queries=z3_queries,
    )
