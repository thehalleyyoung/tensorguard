/-
TensorGuard.Symexec

Aggregator for the machine-checked soundness of the torch-free symbolic-
execution engine (`src/symexec/`).  Importing this module pulls in:

* the abstract-value lattice (`Symexec.Lattice`) and the (α, γ) Galois
  connection (`Symexec.Galois`, Step 93);
* the shared refutation-soundness core (`Symexec.Core`);
* one soundness + certified-witness module per bug detector under
  `Symexec.Transfer.*` (matmul, broadcast, reshape, cat/stack, Linear, unpack
  arity, einsum, axis-OOB, index-OOB, negative dim, div-by-zero, `.item()` on a
  non-scalar, boolean-context on a non-scalar, `.numpy()` on a grad tensor,
  `requires_grad=True` on an integer/bool dtype, and `.backward()` on a
  non-scalar);
* the abstract store with merge/widening soundness (`Symexec.Store`);
* the uniform check framework and small-step semantics (`Symexec.Semantics`,
  Step 91); and
* the whole-program refutation-soundness theorem (`Symexec.Soundness`,
  Step 92): a report implies a real failing concretization exists.
-/

import TensorGuard.Symexec.Lattice
import TensorGuard.Symexec.Galois
import TensorGuard.Symexec.Core
import TensorGuard.Symexec.Store
import TensorGuard.Symexec.Semantics
import TensorGuard.Symexec.Transfer.Matmul
import TensorGuard.Symexec.Transfer.Broadcast
import TensorGuard.Symexec.Transfer.Reshape
import TensorGuard.Symexec.Transfer.CatStack
import TensorGuard.Symexec.Transfer.Linear
import TensorGuard.Symexec.Transfer.UnpackArity
import TensorGuard.Symexec.Transfer.Einsum
import TensorGuard.Symexec.Transfer.AxisOOB
import TensorGuard.Symexec.Transfer.IndexOOB
import TensorGuard.Symexec.Transfer.DivZero
import TensorGuard.Symexec.Transfer.NegativeDim
import TensorGuard.Symexec.Transfer.ItemNonScalar
import TensorGuard.Symexec.Transfer.BoolNonScalar
import TensorGuard.Symexec.Transfer.NumpyOnGrad
import TensorGuard.Symexec.Transfer.RequiresGradNonFloat
import TensorGuard.Symexec.Transfer.BackwardNonScalar
import TensorGuard.Symexec.Transfer.RepeatDimsTooFew
import TensorGuard.Symexec.Transfer.ExpandShapeMismatch
import TensorGuard.Symexec.Transfer.EinopsRankMismatch
import TensorGuard.Symexec.Transfer.NoneDeref
import TensorGuard.Symexec.Affine
import TensorGuard.Symexec.Relational
import TensorGuard.Symexec.Soundness
