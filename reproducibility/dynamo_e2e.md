# TG end-to-end + Dynamo correspondence (smaller-N audit)

Reviewer obligation (round 1, W2/Q2): the headline Track-E table
(`experiments_v5/dynamo_correspondence_v5.json`) records
`signature-trusted` for 16 of 17 modules — the contract was the
documented `forward` signature, not a `verify(M)` result.  This file
documents the complementary smaller-N audit where TensorGuard
end-to-end verifies the class body and Dynamo is then run against the
same contract, giving Theorem 5 a non-vacuous empirical
instantiation.

## Method

For each subject (an `nn.Module` subclass), we

1. call `verify_architecture` on the class body with a symbolic
   shape contract (no surrogate, no signature-trust);
2. *if* TG returns `SAFE` with no shape/grad bugs, instantiate the
   module with a concrete material configuration consistent with
   the contract;
3. run `torch.compile(dynamic=True)` and sample 24 in-contract
   inputs;
4. sample out-of-contract probes (rank, channel, dtype) as positive
   controls.

Driver: `experiments_v5/v8/dynamo_e2e/run_dynamo_e2e.py`.
Pinned versions: torch 2.9.1, torchvision 0.24.1, timm 1.0.26.

## Subjects (selected because TG verifies them end-to-end)

| subject | TG status | TG dur (s) | in-ok / 24 |
|---|---|---|---|
| `torchvision.resnet.BasicBlock` | SAFE | 0.17 | 24/24 |
| `torchvision.resnet.Bottleneck` | SAFE | 0.09 | 24/24 |
| `torchvision.mobilenetv2.InvertedResidual` | SAFE | 0.01 | 24/24 |
| `torchvision.squeezenet.Fire` | SAFE | 0.13 | 24/24 |
| `timm.vit.Block` | SAFE | 0.12 | 24/24 |

Post-compile recompile counts on these subjects are 0–2, all
attributable to the standard dynamic-shape graph specialisation.
Out-of-contract probes trigger the expected violation (runtime
error or recompile) in every case.

## Notes

Two timm vision-transformer constituents (`Mlp`, `Attention`) are
*not* included because TG returns UNSAFE on them (with
`null_dereference` / `division_by_zero` warnings on the
configurable `bias` and `num_heads` paths).  We do not paper over
this: the verifier rejects them honestly.

## Cited from the paper

This audit is referenced in §4.3 of the paper as the
end-to-end-verified row of the Dynamo correspondence audit.
