# TensorGuard for VS Code

Inline, zero-annotation verification of PyTorch `nn.Module`s. As you edit a
Python file, TensorGuard squiggles real shape / broadcast / device / dtype /
phase / gradient bugs, shows inferred tensor shapes on hover, and offers
one-click quick-fixes (e.g. correcting a `Linear`'s `in_features`).

The extension is a thin [language client](https://microsoft.github.io/language-server-protocol/)
over the TensorGuard language server. All analysis runs in the verifier; the
editor only renders results.

## Requirements

- A Python interpreter with TensorGuard importable
  (`python -m pip install "git+https://github.com/thehalleyyoung/tensorguard.git"`),
  reachable as `python` or via the `tensorguard.pythonPath` setting.

## How it works

On activation for a Python document the extension launches:

```
python -m src.lsp_server
```

and speaks LSP over stdio. The server analyses the **unsaved buffer** on every
`didOpen` / `didChange`, so diagnostics update live and clear when the code
becomes correct. See `src/lsp_server.py` and `src/lsp_provider.py` in the
TensorGuard repository; the server handshake is covered by
`tests/test_lsp_server.py`.

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `tensorguard.pythonPath` | `python` | Interpreter used to launch the server. |
| `tensorguard.serverModule` | `src.lsp_server` | Module run with `python -m`. |

## Development

```bash
cd editors/vscode
npm install
# Press F5 in VS Code to launch an Extension Development Host.
```
