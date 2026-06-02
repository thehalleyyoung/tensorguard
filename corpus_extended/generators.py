"""Parameterized generators for the *extended* TensorGuard benchmark corpus.

Steps 1-100 froze a hand-curated 16-case corpus (``real_benchmarks/``). To make
the evaluation paper-grade we need *scale*: many more content-addressed cases
spanning the common shape-error families seen in real PyTorch code. Rather than
hand-write hundreds of files, this module emits parameterized **families** of
idiomatic ``nn.Module`` sources, each labeled ``clean`` (should run / verify
SAFE) or ``buggy`` (should raise at runtime / verify UNSAFE).

Crucially the labels are **not asserted, they are executably validated**:
``corpus_extended/build.py`` instantiates every generated module and runs a real
forward pass with the recorded input shapes, requiring that every ``buggy`` case
raises (with an error message matching its ``expected_error_substring``) and
every ``clean`` case runs without error. A case that does not behave as labeled
is a generator bug and fails the build, so the frozen corpus is ground-truth by
construction.

Each generator yields :class:`Case` records with a deterministic ``id`` so the
materialized corpus and its content hashes are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    label: str  # "clean" | "buggy"
    domain: str  # "shape" | "device" | ...
    source: str
    input_shapes: Dict[str, Tuple[int, ...]]
    expected_error_substring: Optional[str] = None
    provenance_type: str = "derived_pattern"
    seed_url: Optional[str] = None
    note: str = ""


def _src(
    init_lines: List[str],
    forward_lines: List[str],
    imports: Tuple[str, ...] = ("import torch", "import torch.nn as nn"),
) -> str:
    """Assemble a flush-left ``class M(nn.Module)`` source with correct indent.

    Built line-by-line (no textwrap.dedent) so interpolated multi-line bodies
    are always indented consistently regardless of how they were constructed.
    """
    lines: List[str] = list(imports)
    lines += ["", "", "class M(nn.Module):", "    def __init__(self):",
              "        super().__init__()"]
    for ln in init_lines:
        lines.append("        " + ln)
    lines += ["", "    def forward(self, x):"]
    for ln in forward_lines:
        lines.append("        " + ln)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Buggy families (each genuinely raises in real torch)
# --------------------------------------------------------------------------- #
def _buggy_linear_inout() -> List[Case]:
    cases = []
    seed = "https://github.com/pytorch/pytorch/issues/179789"
    for in_dim in (8, 16, 32, 64):
        for mid in (8, 16, 32):
            for delta in (1, -1, 4):
                wrong_in = mid + delta
                if wrong_in < 1 or wrong_in == mid:
                    continue
                src = _src(
                    [f"self.a = nn.Linear({in_dim}, {mid})",
                     f"self.b = nn.Linear({wrong_in}, 4)"],
                    ["return self.b(self.a(x))"],
                    imports=("import torch.nn as nn",),
                )
                cases.append(Case(
                    id=f"buggy_lin_{in_dim}_{mid}_{wrong_in}",
                    family="linear_inout_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (2, in_dim)},
                    expected_error_substring="mat1 and mat2 shapes cannot be multiplied",
                    seed_url=seed,
                    note="Chained Linear whose second layer expects the wrong in_features.",
                ))
    return cases


def _buggy_conv_channel() -> List[Case]:
    cases = []
    seed = "https://github.com/pytorch/pytorch/issues/179931"
    for in_ch in (1, 3):
        for out1 in (8, 16, 32, 64):
            for delta in (1, -2, 8):
                wrong_in2 = out1 + delta
                if wrong_in2 < 1 or wrong_in2 == out1:
                    continue
                src = _src(
                    [f"self.c1 = nn.Conv2d({in_ch}, {out1}, 3, padding=1)",
                     f"self.c2 = nn.Conv2d({wrong_in2}, 16, 3, padding=1)"],
                    ["return self.c2(F.relu(self.c1(x)))"],
                    imports=("import torch.nn as nn",
                             "import torch.nn.functional as F"),
                )
                cases.append(Case(
                    id=f"buggy_conv_{in_ch}_{out1}_{wrong_in2}",
                    family="conv_channel_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (2, in_ch, 16, 16)},
                    expected_error_substring="weight of size",
                    seed_url=seed,
                    note="Second Conv2d declares the wrong in_channels.",
                ))
    return cases


def _buggy_flatten_fc() -> List[Case]:
    cases = []
    seed = "https://github.com/pytorch/pytorch/issues/172739"
    for ch in (4, 8):
        for hw in (8, 16):
            correct = ch * hw * hw
            for delta in (1, -10, 100):
                wrong_fc_in = correct + delta
                if wrong_fc_in < 1 or wrong_fc_in == correct:
                    continue
                src = _src(
                    [f"self.c = nn.Conv2d(3, {ch}, 3, padding=1)",
                     f"self.fc = nn.Linear({wrong_fc_in}, 10)"],
                    ["x = F.relu(self.c(x))",
                     "x = torch.flatten(x, 1)",
                     "return self.fc(x)"],
                    imports=("import torch", "import torch.nn as nn",
                             "import torch.nn.functional as F"),
                )
                cases.append(Case(
                    id=f"buggy_flat_{ch}_{hw}_{wrong_fc_in}",
                    family="flatten_fc_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (2, 3, hw, hw)},
                    expected_error_substring="mat1 and mat2 shapes cannot be multiplied",
                    seed_url=seed,
                    note="FC head input size does not match the flattened conv output.",
                ))
    return cases


def _buggy_matmul_inner() -> List[Case]:
    cases = []
    seed = "https://github.com/pytorch/pytorch/issues/176230"
    for m in (4, 8, 16):
        for k in (8, 16, 32):
            for delta in (1, -1, 3):
                wrong_k = k + delta
                if wrong_k < 1 or wrong_k == k:
                    continue
                src = _src(
                    [f"self.w = nn.Parameter(torch.randn({wrong_k}, 5))"],
                    ["return x @ self.w"],
                    imports=("import torch", "import torch.nn as nn"),
                )
                cases.append(Case(
                    id=f"buggy_matmul_{m}_{k}_{wrong_k}",
                    family="matmul_inner_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (m, k)},
                    expected_error_substring="mat1 and mat2 shapes cannot be multiplied",
                    seed_url=seed,
                    note="x @ W where the inner dimensions disagree.",
                ))
    return cases


def _buggy_cat_dim() -> List[Case]:
    cases = []
    seed = "https://github.com/pytorch/pytorch/issues/175683"
    for in_dim in (8, 16, 32):
        for p in (4, 8, 16):
            for delta in (1, 3, -2):
                q = p + delta
                if q < 1 or q == p:
                    continue
                src = _src(
                    [f"self.a = nn.Linear({in_dim}, {p})",
                     f"self.b = nn.Linear({in_dim}, {q})"],
                    ["return torch.cat([self.a(x), self.b(x)], dim=0)"],
                    imports=("import torch", "import torch.nn as nn"),
                )
                cases.append(Case(
                    id=f"buggy_cat_{in_dim}_{p}_{q}",
                    family="cat_dim_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (2, in_dim)},
                    expected_error_substring="Sizes of tensors must match",
                    seed_url=seed,
                    note="torch.cat along dim 0 of two tensors with differing feature dims.",
                ))
    return cases


def _buggy_add_broadcast() -> List[Case]:
    cases = []
    for in_dim in (8, 16, 32):
        for p in (4, 8, 16):
            for delta in (2, 6, -2):
                q = p + delta
                if q < 1 or q == p:
                    continue
                src = _src(
                    [f"self.a = nn.Linear({in_dim}, {p})",
                     f"self.b = nn.Linear({in_dim}, {q})"],
                    ["return self.a(x) + self.b(x)"],
                    imports=("import torch.nn as nn",),
                )
                cases.append(Case(
                    id=f"buggy_add_{in_dim}_{p}_{q}",
                    family="add_broadcast_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (2, in_dim)},
                    expected_error_substring="must match the size of tensor",
                    seed_url=None, provenance_type="canonical_pattern",
                    note="Elementwise add of two non-broadcastable Linear outputs.",
                ))
    return cases


# --------------------------------------------------------------------------- #
# Clean families (each runs / should verify SAFE)
# --------------------------------------------------------------------------- #
def _clean_mlp() -> List[Case]:
    cases = []
    widths_sets = [
        (64, 32, 10), (128, 64, 10), (256, 128, 64, 10), (784, 256, 10),
        (32, 16, 8, 4), (100, 100, 100, 10), (200, 50, 10), (48, 48, 48, 10),
    ]
    for in_dim in (16, 32, 64, 128):
        for widths in widths_sets:
            dims = [in_dim] + list(widths)
            init = [f"self.fc{i} = nn.Linear({dims[i]}, {dims[i+1]})"
                    for i in range(len(dims) - 1)]
            call = "x"
            for i in range(len(dims) - 1):
                if i < len(dims) - 2:
                    call = f"torch.relu(self.fc{i}({call}))"
                else:
                    call = f"self.fc{i}({call})"
            src = _src(init, [f"return {call}"],
                       imports=("import torch", "import torch.nn as nn"))
            wid = "_".join(str(d) for d in dims)
            cases.append(Case(
                id=f"clean_mlp_{wid}", family="clean_mlp", label="clean",
                domain="shape", source=src, input_shapes={"x": (8, in_dim)},
                provenance_type="canonical_clean",
                note="Idiomatic feed-forward MLP with matching dimensions.",
            ))
    return cases


def _clean_conv() -> List[Case]:
    cases = []
    chan_chains = [
        (3, 16, 32), (3, 8, 16, 32), (1, 8, 16), (3, 32, 64),
        (3, 16, 16, 16), (1, 4, 8, 16),
    ]
    for hw in (8, 16, 32):
        for chans in chan_chains:
            init = [f"self.c{i} = nn.Conv2d({chans[i]}, {chans[i+1]}, 3, padding=1)"
                    for i in range(len(chans) - 1)]
            body = "x"
            for i in range(len(chans) - 1):
                body = f"F.relu(self.c{i}({body}))"
            flat_size = chans[-1] * hw * hw
            init.append(f"self.fc = nn.Linear({flat_size}, 10)")
            fwd = [f"x = {body}", "x = torch.flatten(x, 1)", "return self.fc(x)"]
            src = _src(init, fwd,
                       imports=("import torch", "import torch.nn as nn",
                                "import torch.nn.functional as F"))
            cid = "_".join(str(c) for c in chans)
            cases.append(Case(
                id=f"clean_conv_{hw}_{cid}", family="clean_conv", label="clean",
                domain="shape", source=src,
                input_shapes={"x": (4, chans[0], hw, hw)},
                provenance_type="canonical_clean",
                note="Conv stack with a correctly sized classification head.",
            ))
    return cases


def _clean_norm_mlp() -> List[Case]:
    cases = []
    for in_dim in (32, 64, 128, 256):
        for hidden in (64, 128, 256):
            for norm in ("LayerNorm", "BatchNorm1d"):
                norm_ctor = (f"nn.LayerNorm({hidden})" if norm == "LayerNorm"
                             else f"nn.BatchNorm1d({hidden})")
                init = [f"self.fc1 = nn.Linear({in_dim}, {hidden})",
                        f"self.norm = {norm_ctor}",
                        f"self.fc2 = nn.Linear({hidden}, 10)"]
                fwd = ["x = self.fc1(x)", "x = self.norm(x)",
                       "return self.fc2(torch.relu(x))"]
                src = _src(init, fwd,
                           imports=("import torch", "import torch.nn as nn"))
                cases.append(Case(
                    id=f"clean_norm_{norm.lower()}_{in_dim}_{hidden}",
                    family="clean_norm_mlp", label="clean", domain="shape",
                    source=src, input_shapes={"x": (16, in_dim)},
                    provenance_type="canonical_clean",
                    note=f"MLP with a {norm} layer; dimensions consistent.",
                ))
    return cases


_GENERATORS = [
    _buggy_linear_inout,
    _buggy_conv_channel,
    _buggy_flatten_fc,
    _buggy_matmul_inner,
    _buggy_cat_dim,
    _buggy_add_broadcast,
    _clean_mlp,
    _clean_conv,
    _clean_norm_mlp,
]


def all_cases() -> List[Case]:
    """Return every generated case, sorted by id for deterministic order."""
    cases: List[Case] = []
    for gen in _GENERATORS:
        cases.extend(gen())
    cases.sort(key=lambda c: c.id)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids generated"
    return cases


def family_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for c in all_cases():
        counts[c.family] = counts.get(c.family, 0) + 1
    return dict(sorted(counts.items()))
