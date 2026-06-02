# Community shape-stub registry — governance

TensorGuard can be taught the shape contract of a **third-party layer** so that
models using it are verified *precisely* instead of soundly abstaining. This
directory holds **community-contributed** stubs. To keep the registry auditable
and safe to merge, every contribution is a **declarative manifest** — never
Python code — and is checked by CI before it can land.

## What a manifest is

A manifest is a single JSON object describing one layer class:

```jsonc
{
  "class_name": "Linear8bitLt",          // textual constructor name (no import)
  "kind": "last_dim_linear",             // a vetted, built-in transfer kind
  "spec": {                               // declarative parameters for that kind
    "in_arg": "input_features",
    "out_arg": "output_features",
    "arg_names": ["input_features", "output_features", "bias"]
  },
  "provenance": {                         // mandatory, all fields non-empty
    "author": "you",
    "source_url": "https://github.com/.../modules.py",
    "license": "MIT",
    "reviewed_by": "a-maintainer"
  },
  "conformance": [                        // >= 1 case; CI runs all of them
    {"input": ["batch", 768], "ctor_args": [768, 3072],
     "expect": {"output": ["batch", 3072]}},
    {"input": ["batch", 512], "ctor_args": [768, 3072],
     "expect": {"error_contains": "expects last dim=768, got 512"}}
  ]
}
```

### Supported `kind`s

| kind | contract | required `spec` |
| --- | --- | --- |
| `shape_preserving` | output shape == input shape (norms, dropout, wrappers) | — |
| `last_dim_linear`  | `(*, in) -> (*, out)` on the last dim | `in_arg`, `out_arg`, `arg_names` (optional `defaults`, `out_defaults_to_in`) |

`arg_names` lists the **positional** constructor parameters in order, so a call
like `Linear8bitLt(768, 3072)` binds `input_features=768, output_features=3072`.

## Why declarative-only (security)

Community stubs are **never executable code**. A manifest selects an existing,
already-audited transfer function and supplies parameters. Any manifest that
carries a code-bearing field (`transfer`, `code`, `python`, `eval`, `exec`,
`import`, …) is **rejected**, so reviewing or running a submission can never
execute attacker-controlled code. This is the registry's analogue of the
project's safe-static-analysis guarantee for untrusted models.

## Review process

1. Open a PR adding one `*.json` manifest to `community_stubs/`.
2. CI (`.github/workflows/stub-registry.yml`) runs
   `python -m src.stub_governance_cli --check community_stubs/` which:
   - rejects code-bearing fields,
   - requires complete `provenance` (`author`, `source_url`, `license`,
     `reviewed_by`),
   - registers the declared stub in an **isolated** registry and runs **every**
     conformance case (output shape and/or expected error).
3. A maintainer listed in `MAINTAINERS.md` reviews provenance (correct upstream
   source, compatible license) and approves.
4. On merge, applications opt in at runtime with
   `tensorguard … --community-stubs community_stubs/` (or
   `src.stub_governance.load_community_stubs("community_stubs")`).

A stub that does not behave as its conformance cases claim **cannot merge** —
conformance is proof, not a promise.

## Provenance & license

Only contribute stubs for layers whose upstream source you link and whose
license permits modelling its public shape contract (the manifest contains no
upstream code, only a name and a shape rule). List yourself as `author` and the
reviewing maintainer as `reviewed_by`.
