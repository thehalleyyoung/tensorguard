# Dynamo falsifier-evaluable curated module (round-4 W6 / Q6)

## What this artefact closes

Reviewer round-4 W6 / Q6 asked for at least one curated module
on which the falsification predicate of Theorem~5
(SHAPE/DTYPE/RANK guard outside `catalogue(M)`) is *capable* of
firing, and for confirmation that the necessary direction still
holds there.

## Construction

`reproducibility/dynamo_falsifier_curated_module_fixture.py` defines a single-module
fixture, `ShapeGuardModule(nn.Module)`, that wraps a
`torch.library.custom_op` whose Python kernel branches on
`x.shape[-1]` --- a shape bit *not* declared by any analyser rule
for the surrounding module.  The catalogue $C(M)$ declared by the
analyser is

```
catalogue(M) = { x.shape[0],
                  self.linear.weight.shape[0],
                  self.linear.weight.shape[1] }
```

while the custom op reads `x.shape[-1]`.  By construction this
read is *outside* $C(M)$, so any Dynamo-emitted guard on
`x.shape[-1]` is a SHAPE guard outside $C(M)$ and the
falsification predicate is non-vacuously evaluable.

## Reading

The 14-module CNN-only audit and the 55-module Dynamo audit in
the paper give the in-population evidence for the necessary
direction; both yielded zero falsifier events of the right kind
(13 SHAPE recompiles in-catalogue, 72 INT-only recompiles).
This curated fixture supplies the missing complementary evidence
that the falsification predicate is *capable* of firing on
something of kind ``\ne\ INT`` --- it just doesn't fire in the
typical timm/transformers/torchvision long tail.  The
necessary-direction statement of Theorem~5 still holds on this
module: every analyser refinement variable on `ShapeGuardModule`
is in $C(M)$; the custom op's `x.shape[-1]` read is the
strict-subset witness, not a counterexample.

## Numeric summary

| field | value |
|---|---|
| catalogue(M) size | 3 |
| custom-op shape reads outside catalogue(M) | 1 |
| falsifier-evaluable on event of kind SHAPE | True |
| necessary direction holds | True |

## Paper claim

This artefact is cited by the paragraph following the Audit on a
strictly larger module population paragraph in the eval section,
which had previously stated only that an "adversarial custom op
that reads a shape bit not declared by any rule could in
principle fire the falsifier". This artefact converts that
sentence from "in principle" to "in a checked-in fixture".
