# Import / analysis cost benchmarks

TensorGuard verifies a model from its *source* (AST plus Z3) and never imports the deep-learning runtime to do so. This harness measures import and analysis cost in fresh subprocesses and asserts the torch-free invariant. The committed manifest is deterministic (invariants and ceilings only); measured wall-clock cost is enforced live by `make cost-benchmarks-gate`.

| Cost | Ceiling (s) | torch-free |
|------|-------------|-----------|
| import `verify_model` | 2.0 | yes |
| analyze small model | 5.0 | yes |
| analyze medium model | 20.0 | yes |
