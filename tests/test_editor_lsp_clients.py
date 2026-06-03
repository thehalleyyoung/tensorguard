"""Step 271 — JetBrains and Neovim clients over the shared LSP provider.

These tests are intentionally structural and torch-free: they prove the editor
clients are transport shims over ``src.lsp_server`` and advertise diagnostics,
hovers, and quick-fixes without reimplementing TensorGuard verification logic.
The provider/server behavior itself is covered by ``test_lsp_provider.py`` and
``test_lsp_server.py`` against real verifier output.
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_neovim_client_attaches_python_buffers_to_shared_lsp_server():
    lua = _read("editors/neovim/lua/tensorguard.lua")
    readme = _read("editors/neovim/README.md")

    assert 'pattern = "python"' in lua
    assert "vim.lsp.start" in lua
    assert '"-m", opts.server_module' in lua
    assert 'server_module = "src.lsp_server"' in lua
    assert "make_client_capabilities" in lua
    assert "vim.lsp.buf.hover" in lua
    assert "vim.lsp.buf.code_action" in lua
    assert "diagnostics, hover shapes, and quick-fix code actions" in readme
    assert "verify_architecture" not in lua


def test_jetbrains_plugin_descriptor_registers_lsp_support_provider():
    plugin_xml = ROOT / "editors/jetbrains/resources/META-INF/plugin.xml"
    root = ET.parse(plugin_xml).getroot()
    text = plugin_xml.read_text(encoding="utf-8")

    assert root.findtext("id") == "dev.tensorguard.jetbrains"
    assert "platform.lsp.serverSupportProvider" in text
    assert "TensorGuardLspServerSupportProvider" in text
    assert "com.intellij.modules.platform" in text
    assert "diagnostics, hovers, and quick-fixes" in text


def test_jetbrains_kotlin_launches_shared_server_without_reimplementing_verifier():
    kotlin = _read(
        "editors/jetbrains/src/main/kotlin/dev/tensorguard/jetbrains/"
        "TensorGuardLspServerSupportProvider.kt"
    )
    readme = _read("editors/jetbrains/README.md")

    assert "LspServerSupportProvider" in kotlin
    assert 'file.extension == "py"' in kotlin
    assert 'System.getenv("TENSORGUARD_PYTHON") ?: "python"' in kotlin
    assert 'System.getenv("TENSORGUARD_LSP_MODULE") ?: "src.lsp_server"' in kotlin
    assert "GeneralCommandLine(python, \"-m\", serverModule)" in kotlin
    assert "project.basePath" in kotlin
    assert "verify_architecture" not in kotlin
    assert "src/lsp_provider.py" in readme
    assert "not compiling the plugin" in readme


def test_editor_clients_share_one_protocol_surface_with_vscode():
    files = {
        "vscode": _read("editors/vscode/src/extension.js"),
        "neovim": _read("editors/neovim/lua/tensorguard.lua"),
        "jetbrains": _read(
            "editors/jetbrains/src/main/kotlin/dev/tensorguard/jetbrains/"
            "TensorGuardLspServerSupportProvider.kt"
        ),
    }

    assert all("src.lsp_server" in content for content in files.values())
    for name, content in files.items():
        assert "verify_architecture" not in content, name
