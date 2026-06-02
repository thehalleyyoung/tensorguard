# Security Policy & Threat Model

## Reporting a vulnerability

Please report suspected vulnerabilities privately via GitHub Security Advisories
("Report a vulnerability" on the repository's **Security** tab) rather than a
public issue. We aim to acknowledge reports within five business days and to
ship a fix or mitigation for confirmed, in-scope issues in a timely manner.

When reporting, please include a minimal reproducer, the TensorGuard version
(`tensorguard --version`), and the Python/PyTorch versions.

## Supported versions

TensorGuard follows semantic versioning (see `DEPRECATION_POLICY.md`). Security
fixes target the latest released minor version.

## Threat model

TensorGuard's primary job is to **analyze model source files that may be
untrusted** — e.g., a model file from a pull request, a third-party repository,
a model zoo, or an artifact produced by another tool. The central security
property is therefore:

> **Analyzing a file never executes that file's code.**

### Trust boundaries

| Input | Trust | How it is handled |
| --- | --- | --- |
| Model **source text** (`.py` file or string) | **Untrusted** | Parsed with Python's `ast` module only. Never `import`-ed, `exec`-uted, `eval`-uated, or `compile`+`eval`-ed. |
| `input_shapes` / CLI flags / config | Trusted (operator-supplied) | Plain data; validated and never used to build code. |
| An **already-instantiated** `nn.Module` passed to `verify_module` / `fx`/`dynamo`/`export` extractors | Caller's responsibility | The caller constructed the object; TensorGuard only traces an object it was *given*. Source-level entry points never instantiate untrusted classes. |

### Static-only guarantee

All source-level analysis flows through `verify_architecture` /
`analyze` / `analyze_file` / `quick_check`, which:

1. read the file as **text** (`Path.read_text`, with decoding errors replaced),
2. parse it with `ast.parse` (compiles nothing, runs nothing), and
3. reason over the resulting AST plus refinement types and Z3.

Because the file is never imported or executed, side effects in module-level
code (file writes, `os.system`, network calls, `__import__` of malicious
packages, fork bombs, etc.) **do not run** during analysis. This is enforced by
a regression test (`tests/test_security.py`) that feeds the verifier a source
whose top-level code would create a sentinel file if executed, and asserts the
sentinel is **never created** while verification still completes and reports the
real shape bug.

For untrusted input, prefer the explicit safe entry points in
`src/safe_loader.py`:

```python
from src.safe_loader import verify_file_safely

result = verify_file_safely("untrusted_model.py",
                            input_shapes={"x": ("batch", 10)})
# untrusted_model.py is read as text and analyzed on the AST path;
# its top-level code never runs.
```

### Runtime extractors (out of band)

The `fx`, `dynamo`, and `export` graph extractors operate on an **already
constructed** `nn.Module` instance. Constructing that instance runs the model
author's code and is therefore the **caller's** trust decision — exactly as it
would be for any `import` of that module. TensorGuard's source-level/file
analysis path never constructs such objects from untrusted text, so the runtime
extractors are not reachable from the untrusted-input boundary.

### Out of scope

* Bugs in PyTorch, Z3, or the Python interpreter themselves.
* Code the **caller** chooses to import or instantiate before handing a live
  `nn.Module` to a runtime extractor.
* Denial of service from pathologically large/adversarial inputs to the solver
  (mitigated by `max_cegar_iterations` and solver timeouts, not a security
  guarantee).

## Hardening checklist for integrators

* Verify untrusted files with `verify_file_safely` / `verify_architecture`
  (source-level), **not** by importing them and calling `verify_module`.
* Run the analyzer with least privilege (no write access it doesn't need).
* Treat analyzer output as data; do not feed it back into a shell unquoted.
