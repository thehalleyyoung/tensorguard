# AST-pattern-disabled 60-bug corpus run

## Setup
- Full pipeline: operator-dispatch + flow-sensitive AST-pattern analyser
- Disabled: operator-dispatch only (high_confidence_only=True suppresses the
  parallel flow-sensitive path)

## Results
| Mode | RP / 60 |
|------|---------|
| Full pipeline | 53/60 |
| AST-pattern path disabled | 53/60 |

## Analysis
- Bugs caught by full pipeline but NOT by operator-dispatch alone: 0/60
- These are attributable to the parallel AST-pattern path.
- Operator-dispatch-only contribution: 53/60

## Regressions (full-pipeline caught, disabled missed)
