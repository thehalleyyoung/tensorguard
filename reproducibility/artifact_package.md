# Best-paper artifact package (Step 267)

TensorGuard's artifact is packaged three ways — Docker, conda, and source — and each path has an explicit fresh-machine command, hidden-state barrier, and smoke/evidence command.

- modes checked: **3**
- all modes passed: **True**
- real smoke: `examples/shape_bug.py` -> **UNSAFE** (4 bugs)
- check command: `python reproducibility/artifact_package.py --check`

| mode | fresh-machine command | full evidence command | passed |
| --- | --- | --- | --- |
| docker | `docker build -t tensorguard . && docker run --rm -v "$PWD:/work" tensorguard verify /work/examples/shape_bug.py -s x=1,3,32,32` | `docker build -f capsule/Dockerfile.reproduce -t tensorguard-capsule . && docker run --rm tensorguard-capsule` | True |
| conda | `python -m build --sdist && conda build conda-recipe/ --no-test && conda create -n tg-artifact tensorguard && conda run -n tg-artifact tensorguard --help` | `conda run -n tg-artifact python reproducibility/reproduce_all.py --check` | True |
| source | `git clone https://github.com/thehalleyyoung/tensorguard && cd tensorguard && python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]' && tensorguard verify examples/shape_bug.py -s x=1,3,32,32` | `python reproducibility/reproduce_all.py --check` | True |

## Freshness barriers

### docker

- builds from python:3.12-slim
- runtime stage installs only the built wheel
- runs as an unprivileged user
- image context excludes .git, virtualenvs, caches, and 100_STEPS.md

### conda

- conda-build creates an isolated host/test prefix
- recipe is noarch: python
- runtime dependencies mirror pyproject
- recipe smoke test runs the installed console script

### source

- starts from a fresh clone
- creates a new virtual environment
- installs through pyproject metadata
- runs the same real bug smoke test before evidence replay
