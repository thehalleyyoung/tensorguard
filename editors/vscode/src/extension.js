// TensorGuard VS Code extension (Step 164).
//
// A thin language client that launches the TensorGuard language server
// (`python -m src.lsp_server`) over stdio and wires it to Python documents. The
// server pushes shape/device/dtype/phase/gradient diagnostics, hover shapes, and
// quick-fixes computed by the real verifier — the editor is a transport, not a
// re-implementation.

const { workspace } = require("vscode");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

let client;

function activate(context) {
  const cfg = workspace.getConfiguration("tensorguard");
  const python = cfg.get("pythonPath", "python");
  const serverModule = cfg.get("serverModule", "src.lsp_server");

  const serverOptions = {
    run: { command: python, args: ["-m", serverModule], transport: TransportKind.stdio },
    debug: { command: python, args: ["-m", serverModule], transport: TransportKind.stdio },
  };

  const clientOptions = {
    documentSelector: [{ scheme: "file", language: "python" }],
    synchronize: {
      fileEvents: workspace.createFileSystemWatcher("**/*.py"),
    },
  };

  client = new LanguageClient(
    "tensorguard",
    "TensorGuard",
    serverOptions,
    clientOptions
  );
  client.start();
  context.subscriptions.push(client);
}

function deactivate() {
  return client ? client.stop() : undefined;
}

module.exports = { activate, deactivate };
