# TensorGuard precision/recall regression dashboard

Aggregated headline metrics from the committed evaluation artifacts, gated against `dashboard_baseline.json`. Quality metrics are gated by direction; integrity (corpus-size) metrics must not shrink. See `evaluation/dashboard.py` for the threat model.

| Metric | Kind | Direction | Baseline | Current | Status |
|--------|------|-----------|----------|---------|--------|
| TG precision (real benchmarks) | quality | higher_better | 1.0 | 1.0 | ok |
| TG recall (real benchmarks) | quality | higher_better | 1.0 | 1.0 | ok |
| TG F1 (real benchmarks) | quality | higher_better | 1.0 | 1.0 | ok |
| TG false positives (real benchmarks) | quality | lower_better | 0 | 0 | ok |
| TG false negatives (real benchmarks) | quality | lower_better | 0 | 0 | ok |
| Benchmark population size | integrity | higher_better | 16 | 16 | ok |
| Sound-mode false positives (clean models) | quality | lower_better | 0 | 0 | ok |
| Sound-mode false-positive rate | quality | lower_better | 0.0 | 0.0 | ok |
| Sound-mode clean population size | integrity | higher_better | 80 | 80 | ok |
| Differential-fuzz false positives | quality | lower_better | 0 | 0 | ok |
| Differential-fuzz safe coverage | quality | higher_better | 1.0 | 1.0 | ok |
| Differential-fuzz verified-safe population | integrity | higher_better | 200 | 200 | ok |
| Negative-fuzz recall on injected faults | quality | higher_better | 1.0 | 1.0 | ok |
| Negative-fuzz false negatives | quality | lower_better | 0 | 0 | ok |
| Negative-fuzz genuine-fault population | integrity | higher_better | 281 | 281 | ok |
| Latent-bug recall advantage over baseline | quality | higher_better | 0.75 | 0.75 | ok |
| TG latent-bug recall | quality | higher_better | 0.75 | 0.75 | ok |
| TG latent-bug misses | quality | lower_better | 2 | 2 | ok |
| Triage total disagreements | quality | lower_better | 0 | 0 | ok |
| Frozen regression-suite size | integrity | higher_better | 50 | 50 | ok |
