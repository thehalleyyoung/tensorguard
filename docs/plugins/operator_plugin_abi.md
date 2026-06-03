# TensorGuard operator plugin ABI

TensorGuard has two extension paths for third-party layers:

1. **Community stubs** in `community_stubs/` are declarative JSON and safe for
   untrusted pull requests.
2. **Operator plugins** use this ABI for trusted Python packages that need a real
   executable transfer function.

Plugins are never auto-discovered or auto-imported. A caller must explicitly
import a trusted package, obtain `OperatorTheoryContract` objects, validate them,
and install them:

```python
from src.operator_plugin_abi import install_operator_theories
from my_library.tensorguard_plugin import contracts

install_operator_theories(contracts())
```

## Stable contract

Current ABI: `1.0`. TensorGuard accepts contracts with the same major version.
Minor releases may add optional fields; a future major version may break the ABI.

Each `OperatorTheoryContract` declares:

| Field | Purpose |
| --- | --- |
| `class_name` | Textual layer class name matched by the source frontend. |
| `transfer` | Callable `(TensorShape, params) -> (TensorShape | None, str | None)`. |
| `arg_names` / `defaults` | Constructor binding used to build `params`. |
| `conformance` | Executable examples with expected output shapes or errors. |
| `provenance` | Package, version, source URL, license, author. |
| `security_review` | Human-audited attestations for executable plugin code. |

The transfer must be sound by abstention: return `(None, None)` when it cannot
prove a shape, and return an error only for a violation that is guaranteed for
the declared inputs.

## Security review rules

`SecurityReview` is an attestation, not a sandbox. The real control is the
explicit-import policy: TensorGuard never loads plugin code behind a user's back.
Before installing a plugin, reviewers must confirm:

- no import-time side effects beyond defining contracts;
- no network access;
- no filesystem writes;
- deterministic transfer functions;
- no execution of the analyzed model or untrusted model source.

Conformance validation then executes the transfer only on the plugin's declared
test cases and rejects transfers that crash, drift from expected shapes, or claim
the wrong error.
