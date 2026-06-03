# Stub-authoring quick path

Use community stubs when a third-party layer can be described by an
existing declarative transfer kind; use the plugin ABI only for trusted
packages that need executable transfer code.

## Declarative community stub

1. Add one JSON manifest under `community_stubs/`.
2. Choose a vetted kind such as `shape_preserving` or `last_dim_linear`.
3. Include provenance (`author`, `source_url`, `license`, `reviewed_by`).
4. Add at least one valid conformance case and one invalid/error case when the contract has a refutable precondition.
5. Run `python -m src.stub_governance_cli --check community_stubs/` and `python -m pytest tests/test_stub_governance.py -q`.

The authoritative manifest rules live in `community_stubs/README.md`.

## Trusted operator plugin

Only use `src.operator_plugin_abi.OperatorTheoryContract` for trusted
packages. The contract must be explicit-import only, versioned,
provenance-bearing, security-reviewed, and conformance-tested. The
maintainer-facing ABI docs are `docs/plugins/operator_plugin_abi.md` and
`docs/plugins/third_party_conformance.md`.

## Review invariant

A contribution is not accepted because a shape rule sounds plausible. It
is accepted when the declared conformance cases execute and the generated
proof/confidence metadata remains synchronized with the real registry.
