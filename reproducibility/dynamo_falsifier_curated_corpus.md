# Dynamo falsifier-evaluable curated corpus (N=5)

## What this artefact closes

The round-4 artefact `dynamo_falsifier_curated_module.md` exhibited
a single (N=1) module on which the Theorem~5 falsification predicate
is non-vacuously evaluable.  Round-5 weakness W5 / question Q5 asked
for a small curated corpus (N=5--10) of such modules so that the
falsifier-evaluation cardinality moves from existence to a measured
count.

## Construction

`reproducibility/dynamo_falsifier_curated_corpus_fixture.py` defines
five `nn.Module` subclasses, each of which wraps a
`torch.library.custom_op` whose kernel branches on a shape /
dtype / rank read that is *not* in the analyser-declared
`catalogue(M)` for the surrounding module.  By construction every
Dynamo guard the fixture would emit on the custom op's read sits
outside `catalogue(M)` and is of kind SHAPE / DTYPE / RANK rather
than INT.

## Per-module table

| Module | catalogue(M) | custom op reads | kind | outside C(M)? | necessary direction |
|---|---|---|---|---|---|
| `ShapeGuardModule` | {`x.shape[0]`, `self.linear.weight.shape[0]`, `self.linear.weight.shape[1]`} | `x.shape[-1]` | SHAPE | yes | holds |
| `RankGuardModule` | {`x.shape[0]`, `x.shape[1]`, `self.proj.weight.shape[0]`} | `x.dim()` | RANK | yes | holds |
| `DtypeGuardModule` | {`x.shape[0]`, `self.lin.weight.shape[0]`, `self.lin.weight.shape[1]`} | `x.dtype` | DTYPE | yes | holds |
| `InteriorShapeGuardModule` | {`x.shape[0]`, `self.weight.shape[0]`} | `x.shape[1]` | SHAPE | yes | holds |
| `ProductShapeGuardModule` | {`x.shape[0]`, `self.proj.weight.shape[0]`, `self.proj.weight.shape[1]`} | `x.numel()` | SHAPE | yes | holds |

## Numeric summary

* N = 5
* events of kind != INT : 5/5
* events outside catalogue(M) : 5/5
* respect the necessary direction : 5/5

## Reading

The corpus is curated -- the modules are constructed to stress
the falsification predicate, not sampled from the wild.  The
55-module Dynamo audit in the eval section retains the
in-population reading (72/72 INT-only recompiles, falsifier
never fires); this curated corpus moves the falsifier-evaluation
from `0/0` (vacuous) on the wild surface to `0/5` events that
would falsify Theorem~5 on the curated surface, with the same
necessary direction holding on every row.  Both readings are
consistent with Theorem~5: the necessary direction never
breaks; the falsifier never finds a witness.

## Paper claim

This artefact is cited by the eval section as the N=5 curated
corpus on which the Theorem~5 falsification predicate is
non-vacuously evaluable, replacing the prior single-module
existence proof.  Every row respects the necessary direction.
