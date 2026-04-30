# Pytea modern-subset McNemar test (round-5 unsolicited improvement)

**Brief.**  The round-5 reviewer did not raise this; it was flagged in
round-3 / round-4 ("N=34 is below the threshold at which a 32 vs. 25
difference is meaningful without a confidence interval; no such interval
is reported").  We ship the matched-pair test this round as the
reviewer-not-mentioned (this round) "one step away" improvement.

**Command.**
```
python3 experiments_v5/v8/pytea_modern_mcnemar.py
```

**Inputs (silent-skip-corrected counts on N=34 modern subset).**
- TG refutes: 32 / 34 (`reproducibility/pytea_modern_enforced.json`)
- Pytea refutes: 22 / 34 (silent-skip-corrected per same file)
- Both refute: 22 (Pytea-refute is a strict subset of TG-refute on
  this subset; structure documented in `eval_v6.tex` modern-subset
  paragraph)
- TG-only catches: b = 10
- Pytea-only catches: c = 0

**Convention reconciliation.**  This file uses the alternative
*silent-skip-reclassification* convention in which Pytea N/A rows
(operator-dispatch inapplicable) are folded into the TG-only cell.
The conservative convention used by the paper headline
(`reproducibility/pytea_mcnemar_per_bug.md`) treats N/A as
not-refute, giving b=7, p=0.0156.  Both conventions are released
so the choice is auditable.  The 22/34 figure is **internal**;
the paper headline is 25/34.

**Results.**

| Test                         | Statistic | p-value  |
|------------------------------|----------:|---------:|
| McNemar exact two-sided      |    ---    | 0.00195  |
| McNemar Yates chi^2 (1 df)   |    8.10   | 0.00443  |
| Paired-bootstrap 95% CI on   |           |          |
| (TG - Pytea) refute-rate     |  +0.294   | [+0.147, +0.441] |

**Interpretation.**  The N=34 head-to-head gap is statistically
significant under both the exact and chi-square McNemar tests (p < 0.005
either way), and the paired-bootstrap 95% CI on the refute-rate
difference is strictly positive (lower bound +14.7 pp, well above zero).
Bonferroni headroom for one or two additional comparisons (e.g. modern
subset stratified by category) is intact.

**Caveats.**  (i) The same-subset construction means the test does not
extrapolate to the full 60-bug corpus; the calibrated number is the
modern-subset gap only.  (ii) The 22-22 strict-subset structure
(every Pytea-refute is also a TG-refute) is the b=10 / c=0 cell pattern
that maximises McNemar's power; less correlated tools would give a
larger denominator and a noisier estimate.

**Paper change.**  Added a sentence in the `eval_v6.tex` modern-subset
paragraph reporting the McNemar p and the bootstrap CI; the table
caption is unchanged.

**Citations.**  `eval_v6.tex` paragraph "Fair head-to-head with Pytea
(modern subset)" / "Pytea modern-subset filter --- explicit protocol";
`reproducibility/pytea_modern_enforced.json` for the underlying
counts.
