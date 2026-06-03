#!/usr/bin/env python3
"""Step 289 -- live launch dry-run in a fresh local environment.

The dry-run intentionally executes the public demo path against real code while
remaining deterministic enough to commit.  It creates an isolated temporary
virtualenv, installs this checkout without network dependency resolution, runs
the generated quickstart command, writes a gallery bug into the temporary
workspace, proves that bug is reported UNSAFE, and verifies that every evidence
file named by the generated demo script exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "reproducibility" / "launch_dry_run.json"
OUT_MD = REPO / "reproducibility" / "launch_dry_run.md"
OUTPUTS = (OUT_JSON, OUT_MD)


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _json(rel: str) -> Mapping[str, Any]:
    return json.loads(_read(rel))


def _demo_paths() -> List[str]:
    text = _read("docs/launch/demo_script.md")
    paths = set(re.findall(r"`([^`]+\.(?:py|md|json))`", text))
    for code_span in re.findall(r"`([^`]+)`", text):
        for token in code_span.split():
            cleaned = token.strip(".,:;()[]{}")
            if cleaned.endswith((".py", ".md", ".json")):
                paths.add(cleaned)
    return sorted(paths)


def _path_exists_or_is_current_output(rel: str) -> bool:
    path = REPO / rel
    return path.exists() or path in OUTPUTS


def _run(argv: Sequence[str], *, cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.setdefault("TENSORGUARD_NO_UPDATE_CHECK", "1")
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def _parse_cli_json(output: str) -> Mapping[str, Any]:
    start = output.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in CLI output: {output!r}")
    return json.loads(output[start:])


def _gallery_bug_source() -> Dict[str, Any]:
    manifest = _json("examples/model_gallery.json")
    case = manifest["cases"][0]  # type: ignore[index]
    return {
        "slug": case["slug"],
        "source": case["buggy_source"],
        "shape": ",".join(str(dim) for dim in case["input_shapes"]["x"]),
        "expected_bug": case["caught_bug"],
    }


def _normalize_command(argv: Iterable[str], tmp: Path, venv: Path) -> List[str]:
    normalized: List[str] = []
    for part in argv:
        s = str(part)
        s = s.replace(sys.executable, "$PYTHON")
        s = s.replace(str(venv / "bin" / "python"), "$VENV/bin/python")
        s = s.replace(str(venv / "bin" / "pip"), "$VENV/bin/pip")
        s = s.replace(str(venv / "bin" / "tensorguard"), "$VENV/bin/tensorguard")
        s = s.replace(str(tmp), "$TMP")
        s = s.replace(str(REPO), "$REPO")
        normalized.append(s)
    return normalized


def build_dry_run() -> Dict[str, Any]:
    demo_paths = _demo_paths()
    missing_demo_paths = [path for path in demo_paths if not _path_exists_or_is_current_output(path)]
    gallery = _gallery_bug_source()
    steps: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="tensorguard-launch-dry-run-") as td:
        tmp = Path(td)
        venv = tmp / "venv"
        bug_file = tmp / f"{gallery['slug']}_bug.py"

        create_venv = [sys.executable, "-m", "venv", "--system-site-packages", str(venv)]
        proc = subprocess.run(create_venv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        steps.append(
            {
                "name": "create_fresh_virtualenv",
                "command": _normalize_command(create_venv, tmp, venv),
                "exit_code": proc.returncode,
                "passed": proc.returncode == 0,
            }
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout)

        pip = venv / "bin" / "pip"
        install = [str(pip), "install", "--quiet", "--no-deps", "--editable", str(REPO)]
        proc = _run(install, cwd=tmp, timeout=180)
        steps.append(
            {
                "name": "install_checkout_no_network_resolution",
                "command": _normalize_command(install, tmp, venv),
                "exit_code": proc.returncode,
                "passed": proc.returncode == 0,
            }
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout)

        tg = venv / "bin" / "tensorguard"
        quickstart = [
            str(tg),
            "verify",
            str(REPO / "examples" / "quickstart.py"),
            "--input-shape",
            "x=1,3,224,224",
            "--format",
            "json",
            "--no-color",
            "--no-config",
        ]
        proc = _run(quickstart, cwd=tmp)
        quick_json = _parse_cli_json(proc.stdout) if proc.returncode == 0 else {}
        steps.append(
            {
                "name": "run_quickstart_from_generated_demo",
                "command": _normalize_command(quickstart, tmp, venv),
                "exit_code": proc.returncode,
                "passed": proc.returncode == 0 and quick_json.get("verdict") == "SAFE",
                "verdict": quick_json.get("verdict"),
                "bug_count": len(quick_json.get("bugs", [])),
            }
        )

        bug_file.write_text(str(gallery["source"]), encoding="utf-8")
        gallery_bug = [
            str(tg),
            "verify",
            str(bug_file),
            "--input-shape",
            f"x={gallery['shape']}",
            "--format",
            "json",
            "--no-color",
            "--no-config",
        ]
        proc = _run(gallery_bug, cwd=tmp)
        bug_json = _parse_cli_json(proc.stdout)
        steps.append(
            {
                "name": "run_gallery_bug_variant",
                "command": _normalize_command(gallery_bug, tmp, venv),
                "exit_code": proc.returncode,
                "passed": proc.returncode == 1 and bug_json.get("verdict") == "UNSAFE" and bool(bug_json.get("bugs")),
                "verdict": bug_json.get("verdict"),
                "bug_count": len(bug_json.get("bugs", [])),
                "expected_bug": gallery["expected_bug"],
            }
        )

    steps.append(
        {
            "name": "open_evidence_tour_files",
            "command": ["test", "-e", "<each path cited by docs/launch/demo_script.md>"],
            "exit_code": 0 if not missing_demo_paths else 1,
            "passed": not missing_demo_paths,
            "checked_paths": demo_paths,
            "missing_paths": missing_demo_paths,
        }
    )

    return {
        "schema": "tensorguard.launch_dry_run/v1",
        "step": 289,
        "source_demo_script": "docs/launch/demo_script.md",
        "environment": {
            "fresh_workspace": "tempfile.TemporaryDirectory",
            "virtualenv": "python -m venv --system-site-packages",
            "install": "pip install --no-deps --editable $REPO",
            "network_dependency_resolution": False,
        },
        "summary": {
            "step_count": len(steps),
            "all_steps_passed": all(step["passed"] for step in steps),
            "quickstart_verdict": steps[2].get("verdict"),
            "gallery_bug_verdict": steps[3].get("verdict"),
            "demo_paths_checked": len(demo_paths),
        },
        "steps": steps,
    }


def render_markdown(data: Mapping[str, Any]) -> str:
    summary = data["summary"]  # type: ignore[index]
    env = data["environment"]  # type: ignore[index]
    lines = [
        "# Launch dry-run evidence",
        "",
        "This artifact is generated by `reproducibility/launch_dry_run.py`. It",
        "executes the public launch demo path in a fresh temporary workspace and",
        "isolated virtualenv, then records only deterministic pass/fail facts.",
        "",
        f"- source demo script: `{data['source_demo_script']}`",
        f"- fresh workspace: `{env['fresh_workspace']}`",
        f"- virtualenv: `{env['virtualenv']}`",
        f"- install mode: `{env['install']}`",
        f"- network dependency resolution: **{env['network_dependency_resolution']}**",
        f"- all steps passed: **{summary['all_steps_passed']}**",
        f"- quickstart verdict: **{summary['quickstart_verdict']}**",
        f"- gallery bug verdict: **{summary['gallery_bug_verdict']}**",
        "",
        "| Step | Command | Exit | Verdict | Bugs | Passed |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for step in data["steps"]:  # type: ignore[index]
        command = " ".join(step["command"])
        verdict = step.get("verdict", "-")
        bugs = step.get("bug_count", "-")
        lines.append(
            f"| {step['name']} | `{command}` | {step['exit_code']} | {verdict} | {bugs} | {step['passed']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs() -> Dict[str, Any]:
    data = build_dry_run()
    OUT_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(data), encoding="utf-8")
    return data


def _snapshots(paths: Iterable[Path]) -> Dict[Path, str | None]:
    return {path: path.read_text(encoding="utf-8") if path.exists() else None for path in paths}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if launch dry-run artifacts are stale")
    args = parser.parse_args(argv)

    before = _snapshots(OUTPUTS) if args.check else {}
    data = write_outputs()
    if not data["summary"]["all_steps_passed"]:
        print("launch dry-run failed", file=sys.stderr)
        return 1
    if args.check:
        after = _snapshots(OUTPUTS)
        if before != after:
            print("launch dry-run artifacts are stale", file=sys.stderr)
            return 1
    print("launch dry-run: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
