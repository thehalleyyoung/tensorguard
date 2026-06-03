package dev.tensorguard.jetbrains

import com.intellij.execution.configurations.GeneralCommandLine
import com.intellij.openapi.project.Project
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.platform.lsp.api.LspServerDescriptor
import com.intellij.platform.lsp.api.LspServerSupportProvider

class TensorGuardLspServerSupportProvider : LspServerSupportProvider {
    override fun fileOpened(
        project: Project,
        file: VirtualFile,
        serverStarter: LspServerSupportProvider.LspServerStarter,
    ) {
        if (file.extension == "py") {
            serverStarter.ensureServerStarted(TensorGuardLspServerDescriptor(project))
        }
    }
}

private class TensorGuardLspServerDescriptor(project: Project) :
    LspServerDescriptor(project, "TensorGuard") {
    override fun isSupportedFile(file: VirtualFile): Boolean = file.extension == "py"

    override fun createCommandLine(): GeneralCommandLine {
        val python = System.getenv("TENSORGUARD_PYTHON") ?: "python"
        val serverModule = System.getenv("TENSORGUARD_LSP_MODULE") ?: "src.lsp_server"
        val command = GeneralCommandLine(python, "-m", serverModule)
        project.basePath?.let { command.withWorkDirectory(it) }
        return command
    }
}
