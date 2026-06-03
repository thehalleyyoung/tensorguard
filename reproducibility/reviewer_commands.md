# Reviewer reproduction commands (Step 126)

Two shell entrypoints expose the paper's main evidence without requiring a reviewer to inspect the Makefile:

- full reproduction + byte check: `bash scripts/reproduce_main_results.sh`
- check-only wrapper: `bash scripts/check_main_results.sh`
- command preview: `python reproducibility/reviewer_commands.py --dry-run`

The authoritative full command is `python reproducibility/reproduce_all.py --check`. The manifest covers **14** main result groups; commands resolve: **True**; outputs present: **True**.

| result | claim | command | outputs present |
| --- | --- | --- | --- |
| headline_60bug | 60-bug headline Refuted-Proof figure and README ratios | `python reproducibility/reproduce_headline_60bug.py` | True (reproducibility/reproduce_headline_60bug.json) |
| precision_recall | baseline precision/recall matrices and NA handling | `make precision-recall` | True (evaluation/confusion_matrices.json, evaluation/confusion_matrices.md) |
| significance | McNemar, Holm, and paired-bootstrap significance tests | `python evaluation/significance.py` | True (evaluation/significance.json, evaluation/significance.md) |
| sound_mode_fp | 0% false-positive hunt on executing clean models | `make sound-fp` | True (evaluation/sound_mode_fp.json, evaluation/sound_mode_fp.md) |
| hard_recall | latent-bug recall advantage over the strongest runtime baseline | `make hard-recall` | True (evaluation/hard_recall.json, evaluation/hard_recall.md) |
| differential_fuzz | random valid-module false-positive fuzzing | `make diff-fuzz` | True (evaluation/diff_fuzz.json, evaluation/diff_fuzz.md) |
| negative_fuzz | fault-injection false-negative fuzzing | `make neg-fuzz` | True (evaluation/neg_fuzz.json, evaluation/neg_fuzz.md) |
| triage_regressions | 50 minimized bug reproducers and clean siblings | `make triage` | True (evaluation/triage_regressions.json, evaluation/triage_regressions.md) |
| operator_coverage | public operator coverage matrix | `make operator-coverage` | True (evaluation/operator_coverage.json, evaluation/operator_coverage.md) |
| real_model_operator_coverage | torchvision/timm/HuggingFace frequency-weighted operator coverage | `make real-model-operator-coverage` | True (evaluation/real_model_operator_coverage.json, evaluation/real_model_operator_coverage.md) |
| deployment_gallery | real-model deployment/export gallery and gates | `make deployment-gallery` | True (evaluation/deployment_gallery.json, evaluation/deployment_gallery.md) |
| pareto_curves | hardware-normalized cost/latency Pareto curves | `make pareto-curves` | True (evaluation/pareto_curves.json, evaluation/pareto_curves.md) |
| paper_evidence | single paper-evidence index of every regenerable table/figure | `make paper-evidence` | True (reproducibility/paper_evidence_index.json, reproducibility/paper_evidence_index.md) |
| artifact_index | tamper-evident SHA-256 ledger of generated artifacts | `make artifact-index` | True (reproducibility/artifact_index.json, reproducibility/artifact_index.md) |

## Wrapper scripts

| script | present | sha256 |
| --- | --- | --- |
| `scripts/reproduce_main_results.sh` | True | `e5a0c38008249199df04cdc0ac5c2357c826d8f15bffbf17adfad9229043b928` |
| `scripts/check_main_results.sh` | True | `4b4908d34d7d8d5e4d6de8eb82762f32090cd5e421da0e95a193d1fdcac400bf` |
