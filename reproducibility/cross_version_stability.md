# Cross-version and environment verdict-stability matrix

TensorGuard is a **static** verifier: it analyses source and never executes the target module's framework libraries. Scoring all **227** extended-corpus cases with `torch` and `torchvision` **blocked from import** yields verdicts byte-identical to the normal run (`True`), so the verdict is independent of installed framework binaries.

- baseline verdict-set SHA-256: `e35a57fa51015bb1...`
- sample+fixture verdict-set SHA-256: `2f0adc57b08315b0...`
- static, no target-library execution: **True**
- overall Step 257 stability gate: **True**

PyTorch version matrix (verdicts on a deterministic corpus sample plus torchvision/CPU/MPS fixtures, fake `torch.__version__` pinned):

| torch version | verdicts match baseline |
| --- | --- |
| 2.1.0 | True |
| 2.2.0 | True |
| 2.3.0 | True |
| 2.4.0 | True |
| 2.5.0 | True |
| 2.6.0 | True |
| 2.7.0 | True |
| 2.8.0 | True |
| 2.9.1 | True |

torchvision version matrix (same sample+fixtures, fake `torchvision.__version__` pinned):

| torchvision version | verdicts match baseline |
| --- | --- |
| 0.16.0 | True |
| 0.17.0 | True |
| 0.18.0 | True |
| 0.19.0 | True |
| 0.20.0 | True |
| 0.21.0 | True |
| 0.22.0 | True |
| 0.23.0 | True |
| 0.24.1 | True |

Required Step 257 axes:

| axis | values | status | evidence |
| --- | --- | --- | --- |
| python | 3.9, 3.10, 3.11, 3.12, 3.13, 3.14 | qualified_by_hash_seed_proof | reproducibility/cross_python_determinism.json |
| pytorch | 2.1.0, 2.2.0, 2.3.0, 2.4.0, 2.5.0, 2.6.0, 2.7.0, 2.8.0, 2.9.1 | executed_fake_version_matrix_plus_blocked_import | full extended corpus scored with target framework imports blocked; deterministic sample plus fixtures scored under fake torch.__version__ modules |
| torchvision | 0.16.0, 0.17.0, 0.18.0, 0.19.0, 0.20.0, 0.21.0, 0.22.0, 0.23.0, 0.24.1 | qualified_by_source_level_transform_fixture | torchvision.transforms.v2 fixture verifies identically with torchvision imports blocked and under fake torchvision.__version__ |
| backend | cuda-less CPU, MPS | qualified_by_static_device_source_fixtures | CPU and MPS device-annotation fixtures have identical verdicts with target framework imports blocked; no tensor execution |
| operating_system | linux, macos | qualified_by_source_only_analysis_and_ci_matrix | verdicts depend only on parsed source and committed analyzer tables, not target framework binaries |

Backend fixture verdicts: CUDA-less CPU `SAFE`, MPS `SAFE`; verdicts match with blocked imports: **True**.

The Python axis reuses the committed cross-Python determinism proof (`9d73d39e72cd513e...`), which shows verdict-set digests are invariant under fixed and random `PYTHONHASHSEED` runs. Full historical wheel, interpreter, backend, and OS matrices are version-qualified by the commands recorded in the JSON artifact rather than overclaimed as locally installed here.
