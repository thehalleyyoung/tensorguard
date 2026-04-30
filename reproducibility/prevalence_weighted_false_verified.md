# Prevalence-weighted false-Verified estimate (backward verifier)

## Obligation

Round-1 reviewer item: the limitations section reports two
separately-quantified numbers about the parameter-sharing-under-
renamed-attribute construct family that the first-order grad
lattice silently misclassifies:

* Regex-detectable prevalence ceiling on training scripts:
  <= 12% (from the 2,908-file sweep of HF transformers/diffusers/
  peft repositories).
* Worst-case construct-family false-Verified rate: 2/8 = 25.0%
  on the targeted runtime harness.

The reviewer asked for a single prevalence-weighted product so a
soundness reader can compose the two into a deployment-side
false-Verified bound.

## Command

    python3 reproducibility/prevalence_weighted_false_verified.py

## Method

Conservative product bound:

    P(false_verified | population)
        <= P(construct_family | population)
         * P(false_verified | construct_family)

Both factors are upper bounds; the product is therefore an upper
bound on the deployment-side false-Verified rate.

## Result

| Population | Prevalence factor | Worst-case factor | Composed upper bound |
|---|---|---|---|
| Training scripts (regex screened) | <= 0.12 | <= 0.25 | <= 3.0% |
| Inference scripts (regex screened) | <= 0.04 | <= 0.25 | <= 1.0% |
| Held-out HF examples/pytorch (1/42) | 0.024 | <= 0.25 | <= 0.60% |

Independent runtime input-side aliasing rate: 0/8 = 0.0 on the
data_ptr() check (an orthogonal cap, not multiplied in).

## Paper claim cited

Limitations section: the prevalence-weighted upper bound on the
training-script population is <= 3.0% (= 0.12 x 0.25); on the
inference-script population <= 1.0%; and on the held-out PyTorch
examples directory <= 0.60%. These are reported alongside the
regex prevalence and worst-case construct-family rate.
