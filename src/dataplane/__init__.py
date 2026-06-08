"""The TensorGuard **data plane** — refinement types, effects, and information
flow over the deep-learning *data* layer (vendored from the DataRefine project).

TensorGuard's native analysis is the **model plane**: it verifies tensor algebra
— shapes, dtypes, devices, broadcasting — the structural contract of *how* tensors
flow through a module.  This subpackage adds the orthogonal, complementary
**data plane**: *which numbers reach the model and what they mean*.  A program can
be perfectly shape-correct yet still wrong about its data — a loss applied to
logits instead of probabilities, a scaler fitted before the train/test split, a
feature that reads a future row, a group that straddles both folds, a join that
silently fans out the label, a non-deterministic DataLoader.  None of these are
shape bugs, so the model-plane verifier cannot see them; all of them are
data-plane *typing or non-interference* violations.

The engine is an abstract interpreter over a refinement product lattice
(value-domain × information-flow × split-origin × role × provenance).  It lowers
PyTorch / pandas / sklearn source to a data-plane dataflow, infers each value's
refinement through operator transfer functions, and emits obligations at sinks
that are discharged by a small structural SMT certifier.  See ``NORTH_STAR.md``
in the package for the full theory.

Relationship to the rest of TensorGuard (what is new / redundant / synergistic):

* **New (5 axes + the engine).** Value-domain (loss vs input domain), temporal
  lookahead, group disjointness, join cardinality, and split-contract checks are
  outside any existing TensorGuard analyzer's vocabulary.  More importantly, the
  *abstract-interpretation engine* — a refinement product lattice with operator
  transfer functions that infers a typed contract for every data value and emits
  z3-discharged obligations at sinks — is a strict generalisation of the flat
  pattern scanners.
* **Partially redundant with ``src/interface_layer/torch_data_misuse.py``.** That
  module (merged from PromptABI) already decides three data-plane classes:
  worker-RNG duplication (≈ this engine's ``sampling`` axis), fit-before-split
  leakage (≈ the ``non_interference`` axis), and drop-last-on-eval (which this
  engine does *not* yet model).  The engine re-expresses the first two as
  obligations over its refinement lattice (so they compose with the five new
  axes and export as proof packets); ``torch_data_misuse`` remains the home of
  the drop-last-on-eval check and its dedicated ``scan-torch-data`` CLI.  Prefer
  :func:`analyze_data_plane` when you want the unified seven-axis sweep.
* **Redundant by design (kept separate on purpose).** This subsystem carries its
  own structural SMT certifier (``certification`` / ``smt_backend``) distinct from
  ``src/smt``.  They are *not* merged because they encode different theories: the
  model plane reasons about shape/broadcast/permutation algebra, the data plane
  about value domains, set disjointness, index causality, and cardinality.  Each
  stays the cheapest sound mechanism for its own question.
* **Synergistic.** Both planes now answer to one tool: :func:`analyze_data_plane`
  reports data-plane bugs as TensorGuard :class:`~src.api.Bug` objects, so a
  single ``tensorguard`` run can surface a shape mismatch *and* a data-leakage
  violation side by side.  The data-plane obligations also export as proof
  packets / TensorGuard refinement contexts (``scanners`` module), feeding the
  same proof-carrying pipeline TensorGuard already uses.
"""

from __future__ import annotations

from .dataplane import (
    DATAPLANE_SCHEMA_VERSION,
    DataPlaneInterpreter,
    DataPlaneObligation,
    DataPlaneReport,
    Refinement,
    analyze_all,
    analyze_path,
    analyze_source,
    analyze_tree,
    infer_refinement,
)
from .integration import (
    analyze_data_plane,
    analyze_data_plane_file,
    analyze_data_plane_tree,
    obligation_to_bug,
    report_to_bugs,
)

__all__ = [
    "DATAPLANE_SCHEMA_VERSION",
    "DataPlaneInterpreter",
    "DataPlaneObligation",
    "DataPlaneReport",
    "Refinement",
    "analyze_all",
    "analyze_path",
    "analyze_source",
    "analyze_tree",
    "infer_refinement",
    "analyze_data_plane",
    "analyze_data_plane_file",
    "analyze_data_plane_tree",
    "obligation_to_bug",
    "report_to_bugs",
]
