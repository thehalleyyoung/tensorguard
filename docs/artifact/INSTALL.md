# TensorGuard — Artifact Evaluation: INSTALL

Two supported paths. The **container path** is the most reproducible and is what
we recommend for the Functional badge; the **source path** is best for the
Reusable badge (exercising the public API).

## Path A — container (recommended, fully pinned)

```bash
# Build the reproducible image (multi-stage; ships only the wheel + z3).
docker build -t tensorguard .

# Smoke test: analyse a buggy example. Exit code is non-zero on a found bug.
docker run --rm -v "$PWD:/work" tensorguard verify /work/examples/shape_bug.py
```

The published image is also available without building:

```bash
docker run --rm ghcr.io/thehalleyyoung/tensorguard --help
```

## Path B — source checkout

```bash
git clone https://github.com/thehalleyyoung/tensorguard
cd tensorguard
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # installs z3-solver, torch (CPU), pytest, coverage, …
```

Verify the install:

```bash
python -c "from tensorguard import verify_architecture; print('ok')"
```

## Reproduce the paper's quantitative claims

A single command regenerates every CI-reproducible artifact and then recomputes
every numeric claim in `README.md` from the freshly regenerated files,
asserting determinism:

```bash
python reproducibility/reproduce_all.py --check
```

Expected tail:

```
Reproduction PASS (...). Numeric audit green; every README x/y and % token
recomputed from regenerated artifacts.
DETERMINISM CHECK PASSED ...
```

To regenerate the env-qualified artifacts too (needs Lean / CUDA / network):

```bash
make reproduce-full
```

The claim-by-claim mapping (which command reproduces which number) is in
`docs/artifact/README.md`.

## Run only the new/affected tests

```bash
python -m pytest tests/test_significance.py tests/test_lean_soundness.py \
                 tests/test_formalization.py -q
```
