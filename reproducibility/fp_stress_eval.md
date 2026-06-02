# False-positive stress test

A dedicated stress corpus of **101** clean models across **10** parametric families, built to provoke false alarms with tricky-but-valid shape, broadcast, reshape, concat and normalisation patterns. A false alarm is an UNSAFE verdict on a model that is clean by construction; abstention is reported separately.

| mode | false alarms | FP rate [95% CI] | abstentions |
| --- | --- | --- | --- |
| sound | 0 | 0.0 [0.0, 0.0366] | 0 |
| balanced | 0 | 0.0 [0.0, 0.0366] | 0 |
| heuristic | 0 | 0.0 [0.0, 0.0366] | 0 |

- zero false alarms in sound mode: **True**
- zero false alarms in every mode: **True**
- corpus has at least one hundred clean models: **True**

Every model executes under eager PyTorch, so any UNSAFE verdict would be a false alarm; none occur in any mode.
