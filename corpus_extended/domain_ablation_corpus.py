"""A labeled, multi-domain bug corpus for per-domain ablation (Step 117).

Each generated module contains *exactly one* injected bug of a known domain, so
that a leave-one-domain-out ablation can attribute recall to individual abstract
domains. Five domains are covered:

  * ``shape``    — a dimension/channel mismatch (the always-on base view);
  * ``dtype``    — an integer tensor fed to a float-parameter layer;
  * ``device``   — a CUDA buffer combined with a CPU input;
  * ``gradient`` — a ``.detach()`` severing the only path to the output;
  * ``phase``    — a train/eval-sensitive layer (BatchNorm/Dropout).

The first four are *verification* domains (TensorGuard refutes them); ``phase`` is
*diagnostic-only* (the shipped checker registers phase structure but does not flip
a model from SAFE to UNSAFE), which this corpus records honestly.

Generation is deterministic given a seed. Every non-phase case is constructed so
that the live torch dispatcher would also reject it (shape/dtype/device) or
silently mis-train it (gradient), i.e. the bugs are genuine, not artefacts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

_IMPORTS = "import torch\nimport torch.nn as nn\n\n"

DOMAINS = ("shape", "dtype", "device", "gradient", "phase")
VERIFICATION_DOMAINS = ("shape", "dtype", "device", "gradient")
DIAGNOSTIC_DOMAINS = ("phase",)

# Message-tag -> domain attribution (with CEGAR disabled each bug has one tag).
TAG_TO_DOMAIN: Dict[str, str] = {
    "SHAPE-INCOMPATIBLE": "shape",
    "DTYPE-ERROR": "dtype",
    "DEVICE-MISMATCH": "device",
    "GRADIENT-BROKEN": "gradient",
    "GRADIENT-OUT-OF-FRAGMENT": "gradient",
    "GRADIENT-VIOLATION": "gradient",
    "PHASE-VIOLATION": "phase",
    "PHASE-ERROR": "phase",
}


@dataclass(frozen=True)
class LabeledCase:
    case_id: str
    domain: str
    source: str
    input_shape: Tuple[int, ...]
    is_verification_domain: bool


def _net(init: str, fwd: str) -> str:
    return (
        _IMPORTS
        + "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + init
        + "    def forward(self, x):\n"
        + fwd
        + "        return x\n"
    )


def _gen_shape(rng: random.Random, i: int) -> LabeledCase:
    if rng.random() < 0.5:
        # Linear in_features mismatch.
        f0 = rng.choice([16, 32, 64, 128])
        mid = rng.choice([8, 16, 32])
        wrong = mid + rng.choice([-3, -1, 1, 5, 7])
        wrong = max(2, wrong)
        if wrong == mid:
            wrong += 1
        init = (
            f"        self.a = nn.Linear({f0}, {mid})\n"
            f"        self.b = nn.Linear({wrong}, 10)\n"
        )
        fwd = "        x = self.a(x)\n        x = self.b(x)\n"
        return LabeledCase(f"shape_{i:03d}", "shape", _net(init, fwd), (4, f0), True)
    # Conv channel mismatch.
    c0 = rng.choice([1, 3, 8])
    mid = rng.choice([4, 8, 16])
    wrong = mid + rng.choice([-2, -1, 1, 3])
    wrong = max(1, wrong)
    if wrong == mid:
        wrong += 1
    side = rng.choice([8, 16])
    init = (
        f"        self.c = nn.Conv2d({c0}, {mid}, 3, padding=1)\n"
        f"        self.d = nn.Conv2d({wrong}, 4, 3, padding=1)\n"
    )
    fwd = "        x = self.c(x)\n        x = self.d(x)\n"
    return LabeledCase(f"shape_{i:03d}", "shape", _net(init, fwd), (2, c0, side, side), True)


def _gen_dtype(rng: random.Random, i: int) -> LabeledCase:
    cast = rng.choice(["long", "int", "bool"])
    if rng.random() < 0.5:
        f0 = rng.choice([16, 32, 64])
        mid = rng.choice([8, 16, 32])
        init = f"        self.a = nn.Linear({f0}, {mid})\n"
        fwd = f"        x = x.{cast}()\n        x = self.a(x)\n"
        return LabeledCase(f"dtype_{i:03d}", "dtype", _net(init, fwd), (4, f0), True)
    c0 = rng.choice([1, 3, 8])
    mid = rng.choice([4, 8, 16])
    side = rng.choice([8, 16])
    init = f"        self.c = nn.Conv2d({c0}, {mid}, 3, padding=1)\n"
    fwd = f"        x = x.{cast}()\n        x = self.c(x)\n"
    return LabeledCase(f"dtype_{i:03d}", "dtype", _net(init, fwd), (2, c0, side, side), True)


def _gen_device(rng: random.Random, i: int) -> LabeledCase:
    if rng.random() < 0.5:
        f0 = rng.choice([8, 16, 32])
        init = f"        self.register_buffer('b', torch.zeros({f0}, device='cuda'))\n"
        fwd = "        x = x + self.b\n"
        return LabeledCase(f"device_{i:03d}", "device", _net(init, fwd), (4, f0), True)
    c0 = rng.choice([1, 3, 8])
    side = rng.choice([8, 16])
    init = (
        f"        self.register_buffer('b', "
        f"torch.zeros({c0}, {side}, {side}, device='cuda'))\n"
    )
    fwd = "        x = x + self.b\n"
    return LabeledCase(
        f"device_{i:03d}", "device", _net(init, fwd), (2, c0, side, side), True
    )


def _gen_gradient(rng: random.Random, i: int) -> LabeledCase:
    add = rng.choice([1, 2, 3])
    if rng.random() < 0.5:
        f0 = rng.choice([8, 16, 32])
        init = f"        self.lin = nn.Linear({f0}, {f0})\n"
        fwd = f"        x = self.lin(x).detach() + {add}\n"
        return LabeledCase(f"grad_{i:03d}", "gradient", _net(init, fwd), (4, f0), True)
    c0 = rng.choice([1, 3, 8])
    mid = rng.choice([4, 8, 16])
    side = rng.choice([8, 16])
    init = f"        self.c = nn.Conv2d({c0}, {mid}, 3, padding=1)\n"
    fwd = f"        x = self.c(x).detach() + {add}\n"
    return LabeledCase(
        f"grad_{i:03d}", "gradient", _net(init, fwd), (2, c0, side, side), True
    )


def _gen_phase(rng: random.Random, i: int) -> LabeledCase:
    f0 = rng.choice([8, 16, 32])
    p = rng.choice([0.1, 0.3, 0.5])
    init = (
        f"        self.bn = nn.BatchNorm1d({f0})\n"
        f"        self.drop = nn.Dropout({p})\n"
    )
    fwd = "        x = self.drop(self.bn(x))\n"
    return LabeledCase(f"phase_{i:03d}", "phase", _net(init, fwd), (4, f0), False)


_GENERATORS = {
    "shape": _gen_shape,
    "dtype": _gen_dtype,
    "device": _gen_device,
    "gradient": _gen_gradient,
    "phase": _gen_phase,
}


def build_corpus(seed: int, n_per_domain: int) -> List[LabeledCase]:
    """Build a deterministic labeled corpus, ``n_per_domain`` cases per domain."""

    rng = random.Random(seed)
    cases: List[LabeledCase] = []
    for domain in DOMAINS:
        gen = _GENERATORS[domain]
        for i in range(n_per_domain):
            cases.append(gen(rng, i))
    return cases
