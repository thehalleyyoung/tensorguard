# Full-128 CV joint-satisfiability audit (Round 2 W3/Q1)

## Command
```
python3.11 reproducibility/cv_caller_rely_joint_sat_full128.py
```

## Headline (full 128 CV denominator)

- Witnessed: **118/128** (92.2%)
- Clopper-Pearson 95% CI: **[86.1%, 96.2%]**

## Attempted-only denominator

- Witnessed: **118/128** (92.2%)
- Clopper-Pearson 95% CI: **[86.1%, 96.2%]**

## Bucket breakdown

| bucket | count | jointly satisfied |
|---|---:|---:|
| empty (assume_M ≡ true) | 26 | 26 |
| no-own-init (vacuous) | 12 | 12 |
| symbolic-config-only (attempted) | 90 | 80 |
| symbolic-config-only (excluded) | 0 | 0 (excluded) |

## Notes

* No random sampling is used; the audit covers all 128 CV rows.  The 30-row subsample audit (`cv_caller_rely_joint_sat.{py,json,md}`) is retained for backwards reference.
* Excluded rows are counted as *not witnessed* in the full-128 denominator, so the published ratio is a lower bound.
* Clopper-Pearson CI computed analytically (continued-fraction regularised incomplete beta function); no scipy dependency.
