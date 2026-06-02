"""Offline-replayable issue miner for corpus candidates (Step 103).

Growing a benchmark by hand does not scale. This module *proposes* new corpus
candidates by mining issue reports -- but does so **offline and deterministically**
over local fixtures (``corpus_extended/issue_fixtures/*.json``), so it is fully
replayable in CI with no network access and no flaky GitHub API calls. Each
fixture is a frozen snapshot of an issue (title, labels, body, a reported error
substring and input shapes).

The pipeline is deliberately conservative and **human-in-the-loop**:

1. **Extract** -- pull the first fenced ``python`` code block from the issue body
   and require that it defines an ``nn.Module`` subclass named ``M``.
2. **Replay** -- actually run the extracted module's ``forward`` with the
   reported input shapes against real PyTorch.
3. **Corroborate** -- only propose a *buggy* candidate if the module genuinely
   raises with the reported error substring (so we never trust an issue's claim;
   we verify it). Issues with no code, no error claim, or an unreproducible claim
   are rejected with a reason.
4. **Gate** -- a corroborated candidate is merely ``proposed``. It is promoted to
   ``accepted`` only if a human has added its ``issue_id`` to
   ``corpus_extended/issue_fixtures/accepted.json``. Nothing enters the corpus
   automatically.

This gives an auditable, reproducible miner whose outputs a human reviews before
they ever influence the evaluation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "issue_fixtures"
ACCEPTED_PATH = FIXTURES_DIR / "accepted.json"

_CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Candidate:
    issue_id: str
    title: str
    status: str  # "proposed" | "accepted" | "rejected"
    reason: str
    label: Optional[str]  # "buggy" | "clean" | None
    source: Optional[str]
    input_shapes: Optional[Dict[str, Tuple[int, ...]]]
    expected_error_substring: Optional[str]
    url_reference: Optional[str]


def _extract_code(body: str) -> Optional[str]:
    m = _CODE_BLOCK.search(body)
    if not m:
        return None
    return m.group(1)


def _defines_module(code: str) -> bool:
    return "class M(" in code and "nn.Module" in code


def _replay(code: str, input_shapes: Dict[str, Tuple[int, ...]]) -> str:
    """Run M().forward with the given shapes; return error text or ''."""
    import torch

    ns: dict = {}
    exec(compile(code, "<issue-candidate>", "exec"), ns)
    module = ns["M"]()
    module.eval()
    args = [torch.randn(*shape) for shape in input_shapes.values()]
    try:
        with torch.no_grad():
            module(*args)
    except Exception as exc:  # noqa: BLE001 - probing for failures
        return f"{type(exc).__name__}: {exc}"
    return ""


def _load_accepted() -> set:
    if ACCEPTED_PATH.exists():
        data = json.loads(ACCEPTED_PATH.read_text())
        return set(data.get("accepted_issue_ids", []))
    return set()


def mine_fixture(fixture: dict, accepted: Optional[set] = None) -> Candidate:
    if accepted is None:
        accepted = _load_accepted()
    iid = fixture["issue_id"]
    title = fixture.get("title", "")
    url = fixture.get("url_reference")
    shapes = {k: tuple(v) for k, v in (fixture.get("input_shapes") or {}).items()}
    claimed_sub = fixture.get("reported_error_substring")

    code = _extract_code(fixture.get("body", ""))
    if code is None:
        return Candidate(iid, title, "rejected", "no python code block",
                         None, None, None, None, url)
    if not _defines_module(code):
        return Candidate(iid, title, "rejected", "no nn.Module named M",
                         None, None, None, None, url)
    if not shapes:
        return Candidate(iid, title, "rejected", "no input shapes provided",
                         None, code, None, None, url)

    error_text = _replay(code, shapes)

    if claimed_sub:
        # The issue claims a bug; corroborate by actually reproducing it.
        if not error_text:
            return Candidate(iid, title, "rejected",
                             "claimed error not reproduced (forward ran clean)",
                             None, code, shapes, claimed_sub, url)
        if claimed_sub not in error_text:
            return Candidate(iid, title, "rejected",
                             "reproduced error does not match claimed substring",
                             None, code, shapes, claimed_sub, url)
        status = "accepted" if iid in accepted else "proposed"
        return Candidate(iid, title, status, "corroborated buggy repro",
                         "buggy", code, shapes, claimed_sub, url)
    else:
        # No error claim: only useful as a clean case if it really runs clean.
        if error_text:
            return Candidate(iid, title, "rejected",
                             "no error claimed but forward raised", None, code,
                             shapes, None, url)
        status = "accepted" if iid in accepted else "proposed"
        return Candidate(iid, title, status, "clean repro runs cleanly",
                         "clean", code, shapes, None, url)


def load_fixtures() -> List[dict]:
    fixtures = []
    for p in sorted(FIXTURES_DIR.glob("fixture_*.json")):
        fixtures.append(json.loads(p.read_text()))
    return fixtures


def mine_all(accepted: Optional[set] = None) -> List[Candidate]:
    if accepted is None:
        accepted = _load_accepted()
    return [mine_fixture(f, accepted) for f in load_fixtures()]
