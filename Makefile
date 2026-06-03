# TensorGuard reproducibility harness.
#
# `make reproduce` regenerates every CI-reproducible artifact (the generated
# spec docs/tables, the frozen benchmark corpus + its audit, and the headline
# 60-bug Refuted-Proof figure) from source, then runs the numeric-claim audit
# which recomputes every x/y ratio and % token in README.md from the freshly
# regenerated artifacts.
#
# Artifacts that require CUDA, HuggingFace downloads, or a Lean toolchain are
# not regenerable in a standard CI box; their committed copies are validated by
# the numeric audit (reported as QUALIFIED_ENV) with the regeneration command
# recorded. Use `make reproduce-full` in an environment that has those tools.

PYTHON ?= python3
export PYTHONPATH := $(CURDIR)

.PHONY: help reproduce reproduce-check reproduce-full docs corpus headline audit precision-recall sound-fp hard-recall diff-fuzz neg-fuzz minimize triage shape-props dashboard dashboard-check dashboard-gate operator-coverage operator-coverage-gate operator-coverage-floor fx-trace-success fx-trace-success-gate frontend-parse-sla frontend-parse-sla-gate latency-budgets latency-budgets-gate deployment-budgets deployment-budgets-gate cost-benchmarks cost-benchmarks-gate regression-bench regression-bench-check regression-bench-gate frontend-reconciliation frontend-reconciliation-gate operator-frequency real-model-operator-coverage paper-evidence test clean-pyc

help:
	@echo "TensorGuard make targets:"
	@echo "  reproduce        Regenerate all CI-reproducible artifacts + run the numeric audit"
	@echo "  reproduce-check  reproduce, then assert byte-identical regeneration (no git diff)"
	@echo "  reproduce-full   reproduce-check + the CUDA/HF/Lean artifacts (needs those toolchains)"
	@echo "  docs             Regenerate only the generated spec docs/tables"
	@echo "  corpus           Regenerate only the frozen benchmark corpus + its audit artifact"
	@echo "  headline         Regenerate only the headline 60-bug Refuted-Proof figure"
	@echo "  precision-recall Regenerate the precision/recall confusion matrices vs baselines (needs node for PyTea)"
	@echo "  sound-fp         Regenerate the sound-mode false-positive hunt over clean, executing models"
	@echo "  hard-recall      Regenerate the latent-bug recall comparison vs the strongest dynamic baseline"
	@echo "  diff-fuzz        Regenerate the differential fuzz false-positive hunt over random valid models"
	@echo "  neg-fuzz         Regenerate the negative-fuzz false-negative hunt (inject faults, assert caught)"
	@echo "  minimize         Regenerate the minimal-reproducer shrinker demo (delta-debug a caught fault)"
	@echo "  triage           Regenerate the disagreement triage + 50-case minimal-reproducer regression suite"
	@echo "  shape-props      Run the Hypothesis property tests over the shape-algebra transfer functions"
	@echo "  dashboard        Regenerate the precision/recall regression dashboard (dashboard.md)"
	@echo "  dashboard-check  Gate the dashboard against the frozen baseline (non-zero on regression)"
	@echo "  dashboard-gate   Verify artifacts are fresh, then gate the dashboard (full merge gate)"
	@echo "  operator-coverage  Regenerate the public torch/nn/functional operator coverage matrix"
	@echo "  operator-frequency Regenerate the frequency-weighted operator census over real models"
	@echo "  real-model-operator-coverage  Regenerate the Step 208 torchvision/timm/HF census"
	@echo "  deployment-budgets Regenerate deployment export/compile latency+memory budgets"
	@echo "  deployment-budgets-gate Gate live deployment latency+memory budgets"
	@echo "  audit            Run the numeric-claim audit over committed artifacts"
	@echo "  paper-evidence   Regenerate every table/figure + the single paper-evidence index"
	@echo "  test             Run the pytest suite"

reproduce:
	$(PYTHON) reproducibility/reproduce_all.py

reproduce-check:
	$(PYTHON) reproducibility/reproduce_all.py --check

reproduce-full: reproduce-check
	@echo "Regenerating environment-qualified artifacts (requires CUDA/HF/Lean)..."
	$(PYTHON) reproducibility/audit_numeric_claims.py --regenerate

docs:
	$(PYTHON) -m src.soundness_contract > SOUNDNESS_CONTRACT.md
	$(PYTHON) -m src.verifiable_fragment > VERIFIABLE_FRAGMENT.md
	$(PYTHON) -m src.operator_confidence > operator_confidence_table.json

corpus:
	$(PYTHON) -m real_benchmarks.build_manifest
	$(PYTHON) -m real_benchmarks.build_audit_artifact

headline:
	$(PYTHON) reproducibility/reproduce_headline_60bug.py

precision-recall:
	$(PYTHON) evaluation/precision_recall.py

sound-fp:
	$(PYTHON) evaluation/sound_mode_fp.py

hard-recall:
	$(PYTHON) evaluation/hard_recall.py

diff-fuzz:
	$(PYTHON) evaluation/diff_fuzz.py

neg-fuzz:
	$(PYTHON) evaluation/neg_fuzz.py

minimize:
	$(PYTHON) evaluation/minimize.py

triage:
	$(PYTHON) evaluation/triage.py

shape-props:
	$(PYTHON) -m pytest tests/test_shape_algebra_properties.py -q

dashboard:
	$(PYTHON) evaluation/dashboard.py

dashboard-check:
	$(PYTHON) evaluation/dashboard.py --check

dashboard-gate:
	$(PYTHON) evaluation/precision_recall.py --check
	$(PYTHON) evaluation/sound_mode_fp.py --check
	$(PYTHON) evaluation/diff_fuzz.py --check
	$(PYTHON) evaluation/neg_fuzz.py --check
	$(PYTHON) evaluation/hard_recall.py --check
	$(PYTHON) evaluation/triage.py --check
	$(PYTHON) evaluation/dashboard.py --check

operator-coverage:
	$(PYTHON) evaluation/operator_coverage.py

operator-coverage-gate:
	$(PYTHON) evaluation/operator_coverage.py --gate

operator-coverage-floor:
	$(PYTHON) evaluation/operator_coverage.py --write-floor

fx-trace-success:
	$(PYTHON) evaluation/fx_trace_success.py

fx-trace-success-gate:
	$(PYTHON) evaluation/fx_trace_success.py --gate

frontend-parse-sla:
	$(PYTHON) evaluation/frontend_parse_sla.py

frontend-parse-sla-gate:
	$(PYTHON) evaluation/frontend_parse_sla.py --gate

latency-budgets:
	$(PYTHON) evaluation/latency_budgets.py

latency-budgets-gate:
	$(PYTHON) evaluation/latency_budgets.py --gate

deployment-budgets:
	$(PYTHON) evaluation/deployment_budgets.py

deployment-budgets-gate:
	$(PYTHON) evaluation/deployment_budgets.py --gate

cost-benchmarks:
	$(PYTHON) evaluation/cost_benchmarks.py

cost-benchmarks-gate:
	$(PYTHON) evaluation/cost_benchmarks.py --gate

regression-bench:
	$(PYTHON) evaluation/regression_bench.py

regression-bench-check:
	$(PYTHON) evaluation/regression_bench.py --check

regression-bench-gate:
	$(PYTHON) evaluation/regression_bench.py --gate

frontend-reconciliation:
	$(PYTHON) evaluation/frontend_reconciliation.py

frontend-reconciliation-gate:
	$(PYTHON) evaluation/frontend_reconciliation.py --gate

operator-frequency:
	$(PYTHON) evaluation/operator_frequency.py

real-model-operator-coverage:
	$(PYTHON) evaluation/real_model_operator_coverage.py

audit:
	$(PYTHON) reproducibility/audit_numeric_claims.py

paper-evidence: reproduce-check
	@echo "Building the single paper-evidence index (all tables/figures)..."
	$(PYTHON) reproducibility/paper_evidence_index.py
	$(PYTHON) reproducibility/paper_evidence_index.py --check
	@echo "Paper evidence regenerated; index at reproducibility/paper_evidence_index.md"

test:
	$(PYTHON) -m pytest tests -q --no-header

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
