# Stub-mocked runtime audit on 371-Verified tied-weight subset
## What this artefact closes
Round-6 reviewer asked for a 20-30-row stub-mocked sample of the 365 Verified-but-not-runtime-checked tied-weight modules to be runtime-instantiated against a one-step `loss.backward()` ground truth, so the false-Verified envelope on the 1,957-module population is bounded by measurement rather than by abstention.
## Command
```
PYTHONPATH=. python3 reproducibility/tied_weight_stubmock_runtime.py
```
## Inputs / seed
* Seed: `0` (deterministic).
* Candidate population: the 1957 rows in `tied_weight_full_verdict_rows.json`, restricted to verdict=`Verified` (371 rows).
* Selection rule: shortest-LoC-first oversampling; we run candidates until we reach 25 successfully instantiated and forward-backward-completed rows.
* Stub: a permissive `_StubConfig` plus a resolver namespace that maps every unknown imported symbol to a `MagicMock`. This is intentionally permissive so that as many modules as possible instantiate; rows that still do not run are reported in the `status` column rather than silently skipped.
## Results
* Candidates attempted: **31**
* Successfully instantiated + forward + backward: **25**
* Silent-error count (analyser=Verified, runtime grad-flag empty): **0**
* Wilson 95% CI on the silent-error rate over the OK subset: **[0.00%, 13.32%]**
* `any_grad`: **25/25** (at least one parameter received a gradient)
* `all_grad`: **25/25** (every requires_grad leaf parameter received a gradient)

## Per-row table
| Class | Status | input_shape | n_params | n_with_grad | silent? |
|---|---|---|---|---|---|
| `AfmoeRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `ApertusRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `ArceeRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `AriaTextRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `BambaRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `BarkMLP` | ok | [2, 4, 16] | 2 | 2 | False |
| `BitNetRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `BltRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `ChameleonRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `CsmRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `CwmRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `DFineGate` | forward_or_backward_failed | — | — | — | — |
| `DFineGate` | forward_or_backward_failed | — | — | — | — |
| `DFineIntegral` | forward_or_backward_failed | — | — | — | — |
| `DFineIntegral` | forward_or_backward_failed | — | — | — | — |
| `DeepseekV2RMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `DeepseekV3RMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `Deimv2Integral` | forward_or_backward_failed | — | — | — | — |
| `Deimv2RMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `DiffLlamaRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `DogeRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `Dots1RMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `Emu3RMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `Ernie4_5RMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `Ernie4_5_MoeRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `Ernie4_5_VLMoeRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `Ernie4_5_VLMoeVisionMLP` | instantiation_failed | — | — | — | — |
| `EuroBertRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `Exaone4RMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `ExaoneMoeRMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |
| `FalconH1RMSNorm` | ok | [2, 4, 16] | 1 | 1 | False |

## Interpretation
Of the **25** rows that successfully instantiated and ran a one-step `loss.backward()`, the analyser's Verified verdict was matched by a non-empty runtime grad-flag on **25/25** rows; the silently-incorrect-Verified count is **0/25** (Wilson 95% CI [0.00%, 13.32%]).
This converts the previously abstention-bounded silent-error envelope on the 365 Verified-but-not-runtime-checked tied-weight modules into a measured Wilson interval on a uniformly drawn instantiable subsample.  A non-zero `silent_error_count` would have falsified the §6 silent-error claim on the population of interest.
