# Third-Party Notices and Redistribution Status

TensorGuard is licensed under the **MIT License** (see `LICENSE`). MIT is a
permissive license that is compatible with PyTorch's **BSD-3-Clause** license,
so TensorGuard can be distributed and used alongside (and, if desired,
contributed into) the PyTorch ecosystem without a license conflict.

## What ships in the distributed package

The published sdist and wheel contain **only** the `src/` package plus
documentation (`README.md`, `GETTING_STARTED.md`, `LIMITATIONS.md`,
`API.md`, `SOUNDNESS_CONTRACT.md`, `VERIFIABLE_FRAGMENT.md`) and the `LICENSE`.
This is enforced by `[tool.setuptools.packages.find] include = ["src*"]` in
`pyproject.toml` together with the `prune`/`exclude` rules in `MANIFEST.in`.

All code under `src/` is original TensorGuard code authored for this project.
It does **not** embed or copy third-party source.

## Development-time references (NOT redistributed)

The repository keeps some third-party material purely as **development-time
references** for benchmarking and evaluation. These are excluded from every
distribution artifact (sdist, wheel, and the Docker build context):

* **`experiments_v5/_pytea_src/`** — a checkout of the PyTea project
  (Seoul National University; MIT-licensed) used *only* as a benchmark corpus.
  `src/loaders/pytea_loader.py` parses these files at evaluation time to build
  verification targets; it contains original parsing code and does not copy
  PyTea's implementation. This tree (including its bundled `node_modules/`) is
  large and is never packaged or shipped.

If you redistribute TensorGuard, you are redistributing only the MIT-licensed
`src/` package and docs above; none of the development-time third-party trees
are included.

## How exclusion is verified

`tests/test_distribution_hygiene.py` asserts that:

* `pyproject.toml` restricts packaging to `src*`;
* `MANIFEST.in` prunes `experiments_v5` and excludes vendored `node_modules`;
* a freshly built sdist contains **zero** `_pytea_src`, `node_modules`, or
  `experiments_v5` paths;
* the project license is MIT (BSD-3-compatible) and this notice documents the
  development-time references.
