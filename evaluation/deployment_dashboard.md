# Deployment release dashboard

This tab tracks quantization, export, compile, and distributed gate outcomes per release. Supported rows are ratcheted by `deployment_dashboard_baseline.json`: a pass cannot silently become a skip/fail, and supported rows cannot disappear.

## Release `0.1.0-dev`

| Surface | Backend | Status | Supported | Required | Gate | Evidence |
|---------|---------|--------|-----------|----------|------|----------|
| compile | `torch.dynamo` | passed | yes | env-qualified | `evaluation.deployment_budgets --gate compile/after` | deployment budget post-compile gates |
| distributed | `fsdp+dtensor-static` | passed | yes | yes | `src.distributed_verification.verify_distributed` | FSDP world-size and parameter-sharding shape smoke |
| export | `torch.export` | passed | yes | env-qualified | `evaluation.deployment_gallery --gate post_export_torch_export` | real-model gallery export gates |
| quant | `torch.ao.quantization` | passed | yes | yes | `src.quantization_verify.verify_quantization_eager` | live calibrated QuantStub -> Linear -> DeQuantStub prepared smoke |

**Summary.** 4/4 supported rows passed; failed=0, skipped=0.
