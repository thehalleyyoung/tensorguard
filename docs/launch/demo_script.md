# Public demo script

## Cold open

Run `tensorguard verify examples/quickstart.py --input-shape
x=1,3,224,224 --format json --no-color` to show a fresh install
certifying the quickstart model, then run the generated gallery bug
variant from `reproducibility/launch_dry_run.md` to show TensorGuard
reporting an UNSAFE model-level contract violation before the
matching PyTorch kernel would fail.

## Evidence tour

- Open `examples/model_gallery.md` for the generated model gallery (4 entries).
- Open `examples/tutorials/README.md` for the executable tutorial set (10 notebooks).
- Open `reproducibility/artifact_index.md` to show the generated artifact hash ledger.
- Open `reproducibility/release_readiness.md` to show the launch gate.
- Open `reproducibility/launch_dry_run.md` to show the fresh-venv demo proof.

## Upstream close

End with `docs/upstream/pytorch_proposal.md`: the ask is an opt-in PyTorch
companion hook, not a breaking default-on checker.
