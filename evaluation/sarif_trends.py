"""Regenerate the Step 272 SARIF Code Scanning trend dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.api import verify_architecture
from src.sarif_codescan import build_sarif, check_code_scanning_requirements
from src.sarif_trend_dashboard import SarifSnapshot, build_trend_dashboard, render_markdown

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "evaluation" / "sarif_trends.json"
OUT_MD = ROOT / "evaluation" / "sarif_trends.md"

LINEAR_BAD = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(10, 20)\n"
    "        self.fc2 = nn.Linear(30, 5)\n"
    "    def forward(self, x):\n"
    "        return self.fc2(self.fc1(x))\n"
)

LINEAR_GOOD = LINEAR_BAD.replace("nn.Linear(30, 5)", "nn.Linear(20, 5)")

CONV_BAD = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.conv1 = nn.Conv2d(3, 8, 3)\n"
    "        self.conv2 = nn.Conv2d(9, 16, 3)\n"
    "    def forward(self, x):\n"
    "        return self.conv2(self.conv1(x))\n"
)


def _sarif(items: List[Tuple[str, str, Dict[str, Tuple]]]) -> Dict:
    results = []
    for uri, source, shapes in items:
        result = verify_architecture(source, input_shapes=shapes, filename=uri)
        results.append((uri, result))
    sarif = build_sarif(results)
    problems = check_code_scanning_requirements(sarif)
    if problems:
        raise AssertionError(problems)
    return sarif


def build_dashboard() -> Dict:
    snapshots = [
        SarifSnapshot(
            "v0.1.0",
            _sarif([("models/linear.py", LINEAR_BAD, {"x": ("batch", 10)})]),
        ),
        SarifSnapshot(
            "v0.1.1",
            _sarif([("models/linear.py", LINEAR_GOOD, {"x": ("batch", 10)})]),
        ),
        SarifSnapshot(
            "v0.1.2",
            _sarif(
                [
                    ("models/linear.py", LINEAR_BAD, {"x": ("batch", 10)}),
                    ("models/conv.py", CONV_BAD, {"x": ("batch", 3, 8, 8)}),
                ]
            ),
        ),
    ]
    dashboard = build_trend_dashboard(snapshots)
    rows = dashboard["releases"]
    assert [r["open_total"] for r in rows] == [1, 0, 2]
    assert [r["opened"] for r in rows] == [1, 0, 2]
    assert [r["closed"] for r in rows] == [0, 1, 0]
    assert [r["recurred"] for r in rows] == [0, 0, 1]
    assert dashboard["summary"]["current_open"] == 2
    assert dashboard["summary"]["recurrence_total"] == 1
    return dashboard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    dashboard = build_dashboard()
    json_text = json.dumps(dashboard, indent=2, sort_keys=True) + "\n"
    md_text = render_markdown(dashboard)
    if args.check:
        if OUT_JSON.read_text(encoding="utf-8") != json_text:
            raise SystemExit(f"{OUT_JSON} is stale; run make sarif-trends")
        if OUT_MD.read_text(encoding="utf-8") != md_text:
            raise SystemExit(f"{OUT_MD} is stale; run make sarif-trends")
    else:
        OUT_JSON.write_text(json_text, encoding="utf-8")
        OUT_MD.write_text(md_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
