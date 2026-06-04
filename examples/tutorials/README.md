# TensorGuard tutorials (Colab-runnable)

Eleven small, self-contained notebooks grouped into eight tracks — shapes,
attention, export, distributed, quantization, stubs, formal certificates, and
notebook magic.
Each runs end-to-end on a free Colab CPU runtime in seconds against the *real*
libraries, and each is executed in CI (`tests/test_tutorial_notebooks.py`) so
they never bit-rot.

| Notebook | What it shows |
| --- | --- |
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Catch a one-line shape bug statically, batch-polymorphically |
| [`02_torch_compile.ipynb`](02_torch_compile.ipynb) | Gate `torch.compile` with `guarded_compile` |
| [`03_onnx_export.ipynb`](03_onnx_export.ipynb) | Gate ONNX export (+ `onnx.checker`) with `guarded_onnx_export` |
| [`04_lightning.ipynb`](04_lightning.ipynb) | Verify the net inside a `LightningModule` before `fit` |
| [`05_ci_precommit.ipynb`](05_ci_precommit.ipynb) | Wire a shape-regression gate into CI / pre-commit |
| [`06_attention.ipynb`](06_attention.ipynb) | Verify SDPA attention shape contracts and catch Q/K/V mismatches |
| [`07_distributed.ipynb`](07_distributed.ipynb) | Check FSDP sharding invariants without launching workers |
| [`08_quantization.ipynb`](08_quantization.ipynb) | Gate quantization calibration and autocast policy |
| [`09_community_stubs.ipynb`](09_community_stubs.ipynb) | Validate/load governed community stubs and enforce their contracts |
| [`10_formal_certificates.ipynb`](10_formal_certificates.ipynb) | Produce and replay formal safety certificates |
| [`11_jupyter_magic.ipynb`](11_jupyter_magic.ipynb) | Use `%load_ext` and `%%tensorguard` to check model cells inline |

The notebooks are generated from a single source of truth,
[`build_notebooks.py`](build_notebooks.py); a diff of that file is the diff of
every notebook. Regenerate with:

```bash
python examples/tutorials/build_notebooks.py
```

`tests/test_tutorial_notebooks.py` asserts the committed notebooks match the
generator (no drift) and executes each one (skipping any whose optional
dependency — e.g. `onnx`, `pytorch_lightning` — is absent).
