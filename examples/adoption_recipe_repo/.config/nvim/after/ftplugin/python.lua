vim.keymap.set("n", "<leader>tg", function()
  vim.cmd("!tensorguard verify % --soundness-mode sound")
end)
