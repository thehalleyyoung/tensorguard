# Contemporary execution-based baseline on the N=34 modern subset

## Question

Round-10 reviewer Weakness 2 / Question 4: the fragment-fair
Pytea head-to-head compares against software whose upstream has
been quiet since April 2022.  Supply a contemporary
execution-based baseline on the same 34-bug subset --- e.g., a
hand-annotated jaxtyping/beartype harness, or
`torch.compile(dynamic=True)` graph-break / recompile guard
counts on the same minimal repros.

## Command

```
python reproducibility/contemporary_baseline_34.py
```

## Inputs / seeds / model

* Modern-subset bug list: `experiments_v5/v8/build_modern_subset.py`
  (the 34 entries with `modern=True`, all with primary operators
  in the Pytea-2022 catalogue).
* Per-bug minimal repros: `experiments_v5/bug_repros/bug_*.py`
  (mixed format: zero-arg `_run()` callables, plus five
  `BuggyModule + INPUT_SHAPES` modules for `bug_003..bug_007`).
* `torch == 2.9.1`, `jaxtyping == 0.3.9`, `beartype == 0.22.9`,
  CPU eager / dynamo backend, `torch.manual_seed(7)` reset
  before every individual probe.

## Result

| Tool | Setting | Catches / 34 |
|---|---|---:|
| TensorGuard | static, class source, no instantiation | **32** |
| Pytea (last upstream commit Apr 2022) | static, class source, no instantiation | 22 |
| `jaxtyping` + `beartype` shape contracts | function-boundary annotations on `_run()` | **0** |
| `torch.compile(dynamic=False)` FakeTensor tracing | requires instantiated module **and** concrete inputs | 34 |
| `torch.compile(dynamic=True)` symbolic tracing | requires instantiated module **and** concrete inputs | 34 |

## Interpretation

* `jaxtyping`+`beartype` operate purely as runtime
  function-boundary contracts.  The minimal repros expose no
  user-supplied function boundary that the bug crosses; in every
  case the underlying torch op raises before the contract can
  fire, so the static-catch rate is `0/34`.  These tools are
  complementary to, not substitutes for, static class-source
  analysis.
* `torch.compile` traces under FakeTensor and surfaces the
  shape error at compile time on every repro
  (`34/34` for both `dynamic=False` and `dynamic=True`).  This
  is the strongest contemporary baseline and exceeds TensorGuard
  on this corpus *when the user can supply both an instantiated
  module and concrete example inputs*.  On the 488-block
  real-source corpus the paper studies, 481/488 blocks lack any
  in-repo instantiation harness, which is the inapplicability
  observation the paper already reports.
* The contemporary head-to-head therefore re-positions the
  contribution: TensorGuard is the only tool of the four that
  operates from class source alone, with no instantiated
  `nn.Module` and no concrete inputs.  Within that operating
  regime it catches `32/34` modern-subset bugs vs. Pytea's
  `22/34`; outside it (when concrete inputs exist) `torch.compile`
  is at parity or better, but is unavailable for the
  natural-distribution real-source case.

## Paper claims this artefact backs

* The contemporary-baseline sentence in the eval section
  ("Contemporary baselines on the modern subset: ...") and the
  baseline column added to the head-to-head table.
* The Limitations paragraph note that, when concrete inputs are
  available, `torch.compile` is a competitive
  shape-error-surfacing tool and the contribution of static
  class-source analysis is the inapplicability gap, not absolute
  catch rate, on the bug corpus.
