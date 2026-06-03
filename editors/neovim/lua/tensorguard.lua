-- TensorGuard Neovim LSP client.
--
-- The verifier lives in `python -m src.lsp_server`; this Lua module only
-- attaches Neovim buffers to that server so diagnostics, hovers, and
-- code-actions come from the shared JSON/LSP provider.

local M = {}

local defaults = {
  name = "tensorguard",
  python = "python",
  server_module = "src.lsp_server",
  default_keymaps = true,
}

local function find_root(bufnr)
  local filename = vim.api.nvim_buf_get_name(bufnr)
  local start = filename ~= "" and vim.fs.dirname(filename) or vim.uv.cwd()
  local marker = vim.fs.find({ "pyproject.toml", ".git" }, {
    path = start,
    upward = true,
  })[1]
  return marker and vim.fs.dirname(marker) or vim.uv.cwd()
end

local function attach_keymaps(group, client_name)
  vim.api.nvim_create_autocmd("LspAttach", {
    group = group,
    callback = function(args)
      local client = vim.lsp.get_client_by_id(args.data.client_id)
      if not client or client.name ~= client_name then
        return
      end
      local opts = { buffer = args.buf, silent = true }
      vim.keymap.set("n", "K", vim.lsp.buf.hover, opts)
      vim.keymap.set("n", "<leader>tq", vim.lsp.buf.code_action, opts)
    end,
  })
end

function M.setup(opts)
  opts = vim.tbl_deep_extend("force", defaults, opts or {})
  local group = vim.api.nvim_create_augroup("TensorGuardLsp", { clear = true })

  vim.api.nvim_create_autocmd("FileType", {
    group = group,
    pattern = "python",
    callback = function(args)
      local root_dir = opts.root_dir or find_root(args.buf)
      vim.lsp.start({
        name = opts.name,
        cmd = { opts.python, "-m", opts.server_module },
        root_dir = root_dir,
        filetypes = { "python" },
        capabilities = vim.lsp.protocol.make_client_capabilities(),
      }, { bufnr = args.buf })
    end,
  })

  if opts.default_keymaps then
    attach_keymaps(group, opts.name)
  end
end

return M
