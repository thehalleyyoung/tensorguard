# Head-to-head baseline comparison

Same-corpus comparison on a deterministic stratified subset of **18** cases (12 buggy, 6 clean) covering 9 families.

| tool | static (no exec) | needs inputs | bugs caught | false alarms |
| --- | --- | --- | --- | --- |
| tensorguard | True | False | 12/12 | 0/6 |
| torch_export_trace | False | True | 12/12 | 0/6 |
| mypy | True | False | 0/12 | 0/6 |

TensorGuard on the **full** extended corpus: 153/153 bugs caught, 0/74 false alarms.

- tools catching every subset bug: tensorguard, torch_export_trace
- of those, static *and* input-free: tensorguard
- TensorGuard is the unique static, input-free, complete tool: **True**
- mypy (general static type checker) catches zero shape bugs: **True**

torch.export can also surface these bugs, but only by instantiating the model, building concrete example inputs, and executing a trace. TensorGuard reaches the same verdicts statically from source and declared shapes alone; mypy, the only other static tool, is blind to tensor shapes.
