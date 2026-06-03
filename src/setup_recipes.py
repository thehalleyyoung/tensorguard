"""One-line adoption recipes for TensorGuard integrations.

The recipes are intentionally generated from one small source of truth so the
README, CLI, and tests cannot drift from the integration files they advertise.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class SetupRecipe:
    target: str
    purpose: str
    one_liner: str
    proof_files: Sequence[str]
    smoke_command: str

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)


_GITHUB_WORKFLOW = (
    "mkdir -p .github/workflows && printf '%s\\n' "
    "'name: TensorGuard' 'on: [pull_request]' 'jobs:' '  verify:' "
    "'    runs-on: ubuntu-latest' '    steps:' '      - uses: actions/checkout@v4' "
    "'      - uses: thehalleyyoung/tensorguard@v0.1.0' "
    "'        with:' '          paths: .' '          soundness-mode: sound' "
    "> .github/workflows/tensorguard.yml"
)

_PRECOMMIT = (
    "printf '%s\\n' 'repos:' '  - repo: https://github.com/thehalleyyoung/tensorguard' "
    "'    rev: v0.1.0' '    hooks:' '      - id: tensorguard' "
    ">> .pre-commit-config.yaml && pre-commit install"
)

_PYTEST = "python -m pytest --tensorguard --tensorguard-soundness-mode=sound"

_NOX = (
    "printf '%s\\n' 'import nox' '@nox.session' 'def tensorguard(session):' "
    "'    session.install(\"tensorguard\")' "
    "'    session.run(\"python\", \"-m\", \"pytest\", \"--tensorguard\")' "
    "> noxfile.py && nox -s tensorguard"
)

_TOX = (
    "printf '%s\\n' '[tox]' 'envlist = tensorguard' '[testenv:tensorguard]' "
    "'deps = tensorguard' 'commands = python -m pytest --tensorguard' "
    "> tox.ini && tox -e tensorguard"
)

_MAKEFILE = (
    "printf '\\n%s\\n\\t%s\\n' 'tensorguard:' "
    "'python -m pytest --tensorguard --tensorguard-soundness-mode=sound' "
    ">> Makefile && make tensorguard"
)

_VSCODE = (
    "mkdir -p .vscode && printf '%s\\n' '{\"version\":\"2.0.0\",\"tasks\":[{\"label\":\"TensorGuard\","
    "\"type\":\"shell\",\"command\":\"tensorguard verify ${file} --soundness-mode sound\","
    "\"problemMatcher\":[]}]}' > .vscode/tasks.json"
)

_JETBRAINS = (
    "mkdir -p .idea/runConfigurations && printf '%s\\n' '<component name=\"ProjectRunConfigurationManager\">"
    "<configuration default=\"false\" name=\"TensorGuard current file\" type=\"ShConfigurationType\">"
    "<option name=\"SCRIPT_TEXT\" value=\"tensorguard verify $FilePath$ --soundness-mode sound\" />"
    "</configuration></component>' > .idea/runConfigurations/TensorGuard.xml"
)

_NEOVIM = (
    "mkdir -p .config/nvim/after/ftplugin && printf '%s\\n' "
    "'vim.keymap.set(\"n\", \"<leader>tg\", function() vim.cmd(\"!tensorguard verify % --soundness-mode sound\") end)' "
    "> .config/nvim/after/ftplugin/python.lua"
)

_JUPYTER = (
    "mkdir -p .ipython/profile_default/startup && printf '%s\\n' "
    "'from tensorguard import verify_architecture as tensorguard_verify_architecture' "
    "> .ipython/profile_default/startup/00-tensorguard.py"
)


def recipes() -> List[SetupRecipe]:
    return [
        SetupRecipe(
            target="github-actions",
            purpose="Annotate pull requests and fail CI on unsafe PyTorch modules.",
            one_liner=_GITHUB_WORKFLOW,
            proof_files=(
                "action.yml",
                ".github/workflows/tensorguard-pr.yml",
                "examples/adoption_recipe_repo/.github/workflows/tensorguard.yml",
            ),
            smoke_command="python -m src.github_action",
        ),
        SetupRecipe(
            target="pre-commit",
            purpose="Block commits that introduce statically refutable tensor bugs.",
            one_liner=_PRECOMMIT,
            proof_files=(
                ".pre-commit-hooks.yml",
                "src/precommit.py",
                "examples/adoption_recipe_repo/.pre-commit-config.yaml",
            ),
            smoke_command="tensorguard-precommit path/to/model.py",
        ),
        SetupRecipe(
            target="pytest",
            purpose="Turn an existing test run into a TensorGuard verification gate.",
            one_liner=_PYTEST,
            proof_files=(
                "src/pytest_tensorguard.py",
                "pyproject.toml",
                "examples/adoption_recipe_repo/pyproject.toml",
            ),
            smoke_command=_PYTEST,
        ),
        SetupRecipe(
            target="nox",
            purpose="Add an isolated nox TensorGuard session.",
            one_liner=_NOX,
            proof_files=("noxfile.py", "examples/adoption_recipe_repo/noxfile.py"),
            smoke_command="nox -s tensorguard",
        ),
        SetupRecipe(
            target="tox",
            purpose="Add an isolated tox TensorGuard environment.",
            one_liner=_TOX,
            proof_files=("tox.ini", "examples/adoption_recipe_repo/tox.ini"),
            smoke_command="tox -e tensorguard",
        ),
        SetupRecipe(
            target="makefile",
            purpose="Expose TensorGuard as a make target for local and CI reuse.",
            one_liner=_MAKEFILE,
            proof_files=("Makefile", "examples/adoption_recipe_repo/Makefile"),
            smoke_command="make tensorguard",
        ),
        SetupRecipe(
            target="vscode",
            purpose="Add a VS Code task that verifies the current Python file.",
            one_liner=_VSCODE,
            proof_files=("examples/adoption_recipe_repo/.vscode/tasks.json",),
            smoke_command="Run Task: TensorGuard",
        ),
        SetupRecipe(
            target="jetbrains",
            purpose="Add a JetBrains run configuration for the current file.",
            one_liner=_JETBRAINS,
            proof_files=(
                "examples/adoption_recipe_repo/.idea/runConfigurations/TensorGuard.xml",
            ),
            smoke_command="Run 'TensorGuard current file'",
        ),
        SetupRecipe(
            target="neovim",
            purpose="Add a Neovim mapping that verifies the current Python buffer.",
            one_liner=_NEOVIM,
            proof_files=(
                "examples/adoption_recipe_repo/.config/nvim/after/ftplugin/python.lua",
            ),
            smoke_command="<leader>tg",
        ),
        SetupRecipe(
            target="jupyter",
            purpose="Expose a TensorGuard verification helper in notebooks.",
            one_liner=_JUPYTER,
            proof_files=(
                "examples/adoption_recipe_repo/.ipython/profile_default/startup/00-tensorguard.py",
            ),
            smoke_command="tensorguard_verify_architecture(Model, input_shapes={...})",
        ),
    ]


def recipe_for(target: str) -> SetupRecipe:
    normalized = target.strip().lower()
    for recipe in recipes():
        if recipe.target == normalized:
            return recipe
    known = ", ".join(recipe.target for recipe in recipes())
    raise KeyError(f"unknown setup recipe {target!r}; expected one of: {known}")


def render_text(selected: Iterable[SetupRecipe]) -> str:
    lines: List[str] = []
    for recipe in selected:
        lines.append(f"{recipe.target}: {recipe.one_liner}")
    return "\n".join(lines) + "\n"


def render_json(selected: Iterable[SetupRecipe]) -> str:
    return json.dumps({"recipes": [r.to_dict() for r in selected]}, indent=2) + "\n"


def validate_recipes(repo_root: pathlib.Path | str) -> List[str]:
    """Return validation errors for recipes whose proof files drifted away."""
    errors: List[str] = []
    root = pathlib.Path(repo_root)
    for recipe in recipes():
        if "\n" in recipe.one_liner:
            errors.append(f"{recipe.target}: recipe is not a one-liner")
        for rel in recipe.proof_files:
            path = root / rel
            if not path.exists():
                errors.append(f"{recipe.target}: missing proof file {rel}")
    return errors


__all__ = [
    "SetupRecipe",
    "recipe_for",
    "recipes",
    "render_json",
    "render_text",
    "validate_recipes",
]
