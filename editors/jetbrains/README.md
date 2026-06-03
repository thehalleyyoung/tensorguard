# TensorGuard for JetBrains IDEs

Structurally validated JetBrains LSP plugin template for TensorGuard. It is a
transport shim over `python -m src.lsp_server`, so diagnostics, hovers, and
quick-fixes are computed by the same `src/lsp_provider.py` used by VS Code and
Neovim.

## Scope

This template targets JetBrains IDE builds that expose the platform LSP API
(for example IntelliJ IDEA Ultimate or PyCharm Professional SDKs). The repo does
not vendor the IntelliJ SDK or Gradle wrapper, so tests validate descriptor
shape and shared-server wiring, not compiling the plugin.

## Configuration

The Kotlin descriptor reads:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TENSORGUARD_PYTHON` | `python` | Interpreter used to launch the LSP server. |
| `TENSORGUARD_LSP_MODULE` | `src.lsp_server` | Module run with `python -m`. |

Open the project at a root where TensorGuard is importable, or set `PYTHONPATH`
for the IDE process. The client registers for Python files and delegates all
diagnostic, hover, and `textDocument/codeAction` behavior to the shared server.
