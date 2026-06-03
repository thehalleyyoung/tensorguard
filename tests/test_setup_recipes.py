"""Step 270 — one-line setup recipes for the adoption surface."""

from __future__ import annotations

import json
import os

from src.cli.main import ReftypeCliApp
from src.setup_recipes import recipes, render_json, render_text, validate_recipes


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_recipe_is_one_line_and_backed_by_real_repo_files():
    errors = validate_recipes(os.path.abspath(REPO))  # pathlib accepts path-like strings
    assert errors == []
    for recipe in recipes():
        assert "\n" not in recipe.one_liner
        assert recipe.one_liner.strip()
        assert recipe.smoke_command.strip()


def test_required_adoption_targets_are_all_present():
    targets = {recipe.target for recipe in recipes()}
    assert {
        "github-actions",
        "pre-commit",
        "pytest",
        "nox",
        "tox",
        "makefile",
        "vscode",
        "jetbrains",
        "neovim",
        "jupyter",
    } <= targets


def test_rendered_recipes_are_copy_paste_commands():
    text = render_text(recipes())
    assert "github-actions:" in text
    assert "pre-commit:" in text
    assert "vscode:" in text
    assert "jupyter:" in text
    assert "python -m pytest --tensorguard" in text
    for line in text.strip().splitlines():
        target, command = line.split(": ", 1)
        assert target
        assert command
        assert "\n" not in command


def test_json_render_exposes_proof_files_and_smoke_commands():
    data = json.loads(render_json(recipes()))
    assert len(data["recipes"]) >= 6
    gha = next(r for r in data["recipes"] if r["target"] == "github-actions")
    assert "action.yml" in gha["proof_files"]
    assert gha["smoke_command"] == "python -m src.github_action"


def test_cli_prints_single_target_recipe(capsys):
    rc = ReftypeCliApp().run(["adoption-recipes", "pytest"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() == (
        "pytest: python -m pytest --tensorguard --tensorguard-soundness-mode=sound"
    )


def test_cli_check_and_json(capsys):
    rc = ReftypeCliApp().run(["adoption-recipes", "--check", "--json", "pre-commit"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert [r["target"] for r in data["recipes"]] == ["pre-commit"]
    assert ".pre-commit-hooks.yml" in data["recipes"][0]["proof_files"]


def test_example_repo_backs_every_adoption_surface():
    example_files = {
        "examples/adoption_recipe_repo/model.py",
        "examples/adoption_recipe_repo/.github/workflows/tensorguard.yml",
        "examples/adoption_recipe_repo/.pre-commit-config.yaml",
        "examples/adoption_recipe_repo/pyproject.toml",
        "examples/adoption_recipe_repo/noxfile.py",
        "examples/adoption_recipe_repo/tox.ini",
        "examples/adoption_recipe_repo/Makefile",
        "examples/adoption_recipe_repo/.vscode/tasks.json",
        "examples/adoption_recipe_repo/.idea/runConfigurations/TensorGuard.xml",
        "examples/adoption_recipe_repo/.config/nvim/after/ftplugin/python.lua",
        "examples/adoption_recipe_repo/.ipython/profile_default/startup/00-tensorguard.py",
    }
    for rel in example_files:
        assert os.path.exists(os.path.join(REPO, rel)), rel
