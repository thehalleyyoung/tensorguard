# Dynamo extended audit: CNN end-to-end + transformer surrogate + 55-module sweep

## Question

Round 1 W3: tighten the scope of Theorem 5 (Dynamo
correspondence, necessary direction).  Separate

1. the **9 CNN-block end-to-end** witness (full instantiation,
   `torch.compile(dynamic=True)` over varied inputs, recompile
   events captured),
2. the **transformer-surrogate** witness (4 subjects audited via
   the documented `forward` signature only, since full
   instantiation of window-partition / positional-encoding
   dispatch exceeds end-to-end constraint solving), and
3. the **55-module sweep** finding (only INT specialisations
   fire, zero SHAPE/DTYPE/RANK guards, so the falsification
   predicate is not non-vacuously exercised on the long-tail
   population).

## Headline counts

### CNN-only end-to-end

| Metric | Value |
|---|---:|
| Subjects | **9** CNN blocks |
| Total recompile events | 13 |
| Average SHAPE recompiles per module | 1.44 |
| Guard-kind breakdown | `{SHAPE: 13, DTYPE: 0, RANK: 0, INT: 0}` |
| Guards on in-catalogue refinement variables | 13/13 |
| Falsifier triggered? | **no** (necessary direction supported) |

### Transformer-surrogate

| Metric | Value |
|---|---:|
| Subjects | 4 transformer blocks (forward-signature surrogate) |
| Method | full instantiation of window-partition / positional dispatch exceeds end-to-end constraint solving; we use the documented forward signature only |
| Evidential scope | surrogate-only; reported as context, not as a primary witness |

### 55-module long-tail sweep

| Metric | Value |
|---|---:|
| Candidate modules     | 107 |
| Warmup completed      | **55** |
| Wall-clock killed (240s) | 35 |
| Warmup failed         | 17 |
| In-contract recompiles | 72 |
| Guard-kind breakdown  | `{SHAPE: 0, DTYPE: 0, RANK: 0, INT: 72}` |
| Falsifier exercised non-vacuously? | **no** (denominator zero on SHAPE/DTYPE/RANK guards) |

## Reading

The CNN end-to-end block is the **primary** test of the
necessary direction: 13 SHAPE recompiles on 9 fully instantiated
modules, all on input-shape refinement variables in the TG
catalogue.  The transformer-surrogate evidence is reported as
context only.  The 55-module sweep is reported as a denominator
audit: it is informative that long-tail `nn.Module`s under
`torch.compile(dynamic=True)` recompile predominantly on
integer/`SymInt` specialisations rather than shape guards, but
this means the necessary-direction falsifier is not
non-vacuously exercised on the 55-module population.  We
therefore treat the CNN-only block as the headline.

## Evidential scope summary

* **Primary**: 9 CNN end-to-end (13 SHAPE recompiles, all in
  catalogue).
* **Context**: 4 transformer blocks (forward-signature
  surrogate).
* **Denominator audit**: 55-module sweep (0 SHAPE/DTYPE/RANK
  guards; INT specialisation only).
* **Necessary direction**: supported on the CNN block.
* **Sufficient direction**: not measured (explicitly disclaimed
  in the Theorem 5 statement).

## Paper claims cited

* Theorem 5 statement (necessary direction).
* Eval section extended end-to-end audit paragraph.
* Eval section "audit on a strictly larger module population".
