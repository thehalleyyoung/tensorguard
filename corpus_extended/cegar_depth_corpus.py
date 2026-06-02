"""Labeled corpus for the CEGAR refinement-depth ablation (Step 119).

The shape-CEGAR loop discovers *implicit input contracts* by accumulating shape
predicates from counterexamples. This corpus isolates the bug class on which the
refinement actually does work -- **infeasible-contract bugs** -- so that the
ablation can measure what extra refinement depth buys.

Two families:

* ``conflict`` (buggy): a single input ``x`` flows into two parameterised layers
  whose required last-dimensions disagree (e.g. ``nn.Linear(768, .)`` and
  ``nn.Linear(512, .)`` both consuming ``x``). No concrete input width can
  satisfy both, so the module *always* raises under eager PyTorch -- a genuine
  bug, asserted at build time by replaying every probed width. The input is left
  symbolic precisely because no concrete width is valid.

* ``clean`` (control): an ordinary feed-forward stack with a concrete input
  width that matches its first layer, so it runs under eager PyTorch and is SAFE
  at every CEGAR depth. Used to prove that raising the refinement budget never
  manufactures a false alarm.

Every case is validated against real PyTorch by ``cegar_depth_validate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

_IMPORTS = "import torch\nimport torch.nn as nn\n"


@dataclass(frozen=True)
class CegarCase:
    id: str
    family: str  # "conflict" | "clean"
    source: str
    input_shapes: Dict[str, tuple]
    genuine: bool  # conflict: always raises; clean: always runs


def _conflict_src(a: int, b: int, n_relu: int) -> str:
    body = ["        h = x"] + ["        h = torch.relu(h)"] * n_relu
    lines = [
        _IMPORTS,
        "class M(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        f"        self.a = nn.Linear({a}, 10)",
        f"        self.b = nn.Linear({b}, 10)",
        "    def forward(self, x):",
        *body,
        "        return self.a(h) + self.b(h)",
    ]
    return "\n".join(lines) + "\n"


def _clean_src(w: int, hidden: int) -> str:
    lines = [
        _IMPORTS,
        "class M(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        f"        self.fc1 = nn.Linear({w}, {hidden})",
        f"        self.fc2 = nn.Linear({hidden}, 10)",
        "    def forward(self, x):",
        "        return self.fc2(torch.relu(self.fc1(x)))",
    ]
    return "\n".join(lines) + "\n"


# Deterministic parameter grids.
_CONFLICT_GRID: List[Tuple[int, int, int]] = [
    (768, 512, 0), (512, 256, 0), (256, 128, 0), (1024, 768, 0),
    (768, 512, 1), (512, 256, 1), (256, 128, 1), (640, 320, 1),
    (768, 512, 2), (512, 256, 2), (300, 400, 2), (96, 128, 2),
    (200, 150, 3), (128, 64, 3), (384, 192, 3), (220, 330, 3),
]

_CLEAN_GRID: List[Tuple[int, int]] = [
    (64, 32), (128, 64), (256, 128), (32, 16),
    (512, 64), (96, 48), (200, 50), (48, 24),
]


def build_corpus() -> List[CegarCase]:
    cases: List[CegarCase] = []
    for i, (a, b, nr) in enumerate(_CONFLICT_GRID):
        cases.append(
            CegarCase(
                id=f"conflict_{i:02d}_{a}_{b}_r{nr}",
                family="conflict",
                source=_conflict_src(a, b, nr),
                input_shapes={"x": ("batch", "feat")},
                genuine=True,
            )
        )
    for i, (w, h) in enumerate(_CLEAN_GRID):
        cases.append(
            CegarCase(
                id=f"clean_{i:02d}_{w}_{h}",
                family="clean",
                source=_clean_src(w, h),
                input_shapes={"x": ("batch", w)},
                genuine=True,
            )
        )
    return cases


_PROBE_WIDTHS = [16, 32, 48, 64, 96, 128, 192, 256, 320, 512, 768, 1024]


def _runs_under_torch(source: str, width: int) -> bool:
    import torch  # local import; only used during validation

    ns: dict = {}
    exec(compile(source, "<cegar_case>", "exec"), ns)
    M = ns["M"]
    try:
        model = M()
        model(torch.zeros(2, width))
        return True
    except Exception:
        return False


def cegar_depth_validate(cases: List[CegarCase]) -> None:
    """Assert every case's ground-truth label against real PyTorch."""
    for c in cases:
        if c.family == "conflict":
            # No probed width may run: the contract is genuinely infeasible.
            assert not any(_runs_under_torch(c.source, w) for w in _PROBE_WIDTHS), (
                f"{c.id}: expected infeasible but some width ran"
            )
        else:
            w = c.input_shapes["x"][1]
            assert _runs_under_torch(c.source, int(w)), (
                f"{c.id}: clean control failed to run at width {w}"
            )
