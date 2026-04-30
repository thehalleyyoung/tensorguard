"""Prevalence-weighted false-Verified estimate for the backward verifier.

Round-1 reviewer item: limconc_v6.tex reports a regex-detectable
prevalence ceiling of <= 12% for the parameter-sharing-under-renamed-
attribute construct family on training scripts and a worst-case false-
Verified rate of 2/8 = 25.0% on the targeted construct family.
The reviewer asked for a single prevalence-weighted bound that
composes both numbers across the deployment population.

The simple composition is:

  P(false_verified | training_script)
    <= P(construct_family | training_script) * P(false_verified | construct_family)

with conservative numerator bounds
  P(construct_family | training_script) <= 0.12  (regex-detectable bound)
  P(false_verified | construct_family) <= 2/8 = 0.25  (worst-case bound)

The product is <= 0.12 * 0.25 = 0.030 = 3.0% upper bound on the
training-script population.

For inference scripts (regex prevalence <= 4%):
  <= 0.04 * 0.25 = 0.010 = 1.0% upper bound.

We also report the runtime data_ptr() bound as a separate metric:
the input-side aliasing rate is 0/8, so the runtime-aliased false
Verified rate is bounded by the regex prevalence times the worst
construct-family rate but the data_ptr proxy independently caps the
input-side population at 0/8.
"""

import json
import pathlib
import time

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    regex_prevalence_train = 0.12  # <= 12%, training scripts
    regex_prevalence_infer = 0.04  # <= 4%, inference scripts
    worst_case_false_verified = 2.0 / 8.0  # 25.0% on the construct family
    held_out_train_subset = 1.0 / 42.0  # 2.4% on the held-out HF examples/

    composed_train = regex_prevalence_train * worst_case_false_verified
    composed_infer = regex_prevalence_infer * worst_case_false_verified
    composed_held_out_train = held_out_train_subset * worst_case_false_verified

    out = {
        "_obligation": "Round-1 reviewer item: compose the regex-detectable prevalence (<= 12%) with the worst-case construct-family false-Verified rate (2/8 = 25.0%) into a single prevalence-weighted bound on the deployment-side false-Verified rate of the backward verifier.",
        "_command": "python3 reproducibility/prevalence_weighted_false_verified.py",
        "_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "regex_prevalence_training_scripts": regex_prevalence_train,
            "regex_prevalence_inference_scripts": regex_prevalence_infer,
            "worst_case_false_verified_construct_family": worst_case_false_verified,
            "held_out_train_subset_prevalence": held_out_train_subset,
            "input_side_data_ptr_aliasing_rate": "0/8 = 0.0 (independent runtime cap)",
        },
        "results": {
            "composed_upper_bound_training_scripts": composed_train,
            "composed_upper_bound_inference_scripts": composed_infer,
            "composed_upper_bound_holdout_pytorch_examples": composed_held_out_train,
            "interpretation_training_scripts": f"<= {composed_train*100:.1f}% of training scripts in the regex-screened population are silently false-Verified by the backward verifier under the worst-case construct family",
            "interpretation_inference_scripts": f"<= {composed_infer*100:.1f}% of inference scripts",
            "interpretation_holdout": f"<= {composed_held_out_train*100:.2f}% of the held-out HF training examples (using the held-out 1/42 prevalence rather than the regex ceiling)",
        },
        "method": "Conservative product bound P(false_verified) <= P(construct_family) * P(false_verified | construct_family). Independent in input populations, monotone in both factors.",
        "paper_claim_cited": "Limitations and Conclusion section: the prevalence-weighted false-Verified bound is reported alongside the regex prevalence and worst-case construct-family rate.",
    }
    out_path = REPO / "reproducibility" / "prevalence_weighted_false_verified.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
