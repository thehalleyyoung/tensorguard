# TensorGuard for Neovim

Thin Neovim client for the TensorGuard language server. It starts
`python -m src.lsp_server` for Python buffers and lets the shared LSP provider
render diagnostics, hover shapes, and quick-fix code actions.

## Setup

Copy `lua/tensorguard.lua` into your Neovim runtime path, then add:

```lua
require("tensorguard").setup({
  python = "python",
  server_module = "src.lsp_server",
})
```

The module attaches on `FileType python`, sets the server root to the nearest
`pyproject.toml`/`.git` parent, and falls back to the current working directory.
That root must make TensorGuard importable; from a source checkout, start Neovim
at the repository root or set `PYTHONPATH` accordingly.

Default mappings are buffer-local after `LspAttach`: `K` for TensorGuard hovers
and `<leader>tq` for quick-fixes. Set `default_keymaps = false` to provide your
own mappings.
