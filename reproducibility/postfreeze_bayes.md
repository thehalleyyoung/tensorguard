# Bayesian analysis of N=15 post-freeze unfiltered sample

One-sided Wilson lower bounds and Bayes factors for the
N=15 unfiltered pre-registered post-freeze real-PR sample.

## One-sided Wilson 95% lower bounds

| Tool | Catches | Rate | One-sided 95% lower | Two-sided 95% CI |
|---|---|---|---|---|
| TG | 5/15 | 33.3% | 17.3% | [15.2%, 58.3%] |
| FakeTensorMode | 2/15 | 13.3% | 4.5% | [3.7%, 37.9%] |
| Pytea | 3/15 | 20.0% | 8.3% | [7.0%, 45.2%] |

## Bayes factors (H1: TG > baseline, H0: TG ≤ baseline; Beta(1,1) prior)

| Comparison | BF₁₀ | Evidence level |
|---|---|---|
| TG vs FakeTensorMode | **8.09** | moderate (3-10) |
| TG vs Pytea | **3.62** | moderate (3-10) |

## Interpretation

At N=15 the posterior mean for TG (34.4%) exceeds those of
FT (15.6%) and Pytea (21.9%), and the Bayes factors indicate
BF(TG>FT)=8.09: the data are 8.1x more consistent with TG's catch rate exceeding FT's than with the reverse. BF(TG>Pytea)=3.62: similarly for Pytea. BF < 3 is 'weak evidence', 3-10 is 'moderate evidence'. On N=15 the data are consistent with a TG advantage but do not constitute strong Bayesian evidence (BF >> 10) for superiority over either baseline.

The one-sided Wilson lower bounds confirm that all three tools' catch
rates are plausibly above zero, but the CIs for TG and Pytea overlap,
consistent with the non-significant Fisher-exact p-values.

## Reproduce

    PYTHONPATH=. python3.11 reproducibility/postfreeze_bayes.py
