"""Held-out *blind* test split for the extended corpus (Step 105).

To rule out overfitting the verifier (or its abstract-domain stubs) to the
development corpus, we freeze a **separate** blind split whose cases are
generated from **disjoint parameter grids** -- different feature widths, channel
counts and deltas than :mod:`corpus_extended.generators`. By construction no
case id can collide with the dev corpus, so the split is genuinely held out.

The split is generated with the same line-based ``_src`` builder and the same
nine families, and every case is **runtime-validated against real PyTorch** at
build time (see :mod:`corpus_extended.blind_build`). The accompanying
``PRE_REGISTRATION.md`` records the hypotheses we commit to *before* scoring the
split; :mod:`reproducibility.blind_split_eval` then scores TensorGuard on the
split exactly once and checks those pre-registered predictions.
"""

from __future__ import annotations

from typing import List

from corpus_extended.generators import Case, _src

# Disjoint grids: none of these (in_dim, mid, delta, ...) tuples appear in the
# development generators, guaranteeing held-out case ids.
_BLIND_TAG = "blind"


def _buggy_linear_inout() -> List[Case]:
    cases = []
    seed = "https://github.com/pytorch/pytorch/issues/179789"
    for in_dim in (10, 20, 48):
        for mid in (12, 24, 48):
            for delta in (2, -3, 5):
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
                    id=f"{_BLIND_TAG}_buggy_lin_{in_dim}_{mid}_{wrong_in}",
                    family="linear_inout_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (2, in_dim)},
                    expected_error_substring="mat1 and mat2 shapes cannot be multiplied",
                    seed_url=seed,
                    note="[blind] Chained Linear with wrong second in_features.",
                ))
    return cases


def _buggy_conv_channel() -> List[Case]:
    cases = []
    seed = "https://github.com/pytorch/pytorch/issues/179931"
    for in_ch in (2, 4):
        for out1 in (12, 24, 48):
            for delta in (3, -4, 5):
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
                    id=f"{_BLIND_TAG}_buggy_conv_{in_ch}_{out1}_{wrong_in2}",
                    family="conv_channel_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (2, in_ch, 12, 12)},
                    expected_error_substring="weight of size",
                    seed_url=seed,
                    note="[blind] Second Conv2d declares the wrong in_channels.",
                ))
    return cases


def _buggy_flatten_fc() -> List[Case]:
    cases = []
    seed = "https://github.com/pytorch/pytorch/issues/172739"
    for ch in (6, 12):
        for hw in (10, 12):
            correct = ch * hw * hw
            for delta in (3, -7, 50):
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
                    id=f"{_BLIND_TAG}_buggy_flat_{ch}_{hw}_{wrong_fc_in}",
                    family="flatten_fc_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (2, 3, hw, hw)},
                    expected_error_substring="mat1 and mat2 shapes cannot be multiplied",
                    seed_url=seed,
                    note="[blind] FC head size does not match flattened conv output.",
                ))
    return cases


def _buggy_matmul_inner() -> List[Case]:
    cases = []
    seed = "https://github.com/pytorch/pytorch/issues/176230"
    for m in (6, 12, 20):
        for k in (12, 24, 40):
            for delta in (2, -2, 5):
                wrong_k = k + delta
                if wrong_k < 1 or wrong_k == k:
                    continue
                src = _src(
                    [f"self.w = nn.Parameter(torch.randn({wrong_k}, 5))"],
                    ["return x @ self.w"],
                    imports=("import torch", "import torch.nn as nn"),
                )
                cases.append(Case(
                    id=f"{_BLIND_TAG}_buggy_matmul_{m}_{k}_{wrong_k}",
                    family="matmul_inner_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (m, k)},
                    expected_error_substring="mat1 and mat2 shapes cannot be multiplied",
                    seed_url=seed,
                    note="[blind] x @ W where inner dimensions disagree.",
                ))
    return cases


def _buggy_cat_dim() -> List[Case]:
    cases = []
    seed = "https://github.com/pytorch/pytorch/issues/175683"
    for in_dim in (12, 20, 40):
        for p in (6, 12, 20):
            for delta in (2, 5, -3):
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
                    id=f"{_BLIND_TAG}_buggy_cat_{in_dim}_{p}_{q}",
                    family="cat_dim_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (2, in_dim)},
                    expected_error_substring="Sizes of tensors must match",
                    seed_url=seed,
                    note="[blind] torch.cat dim 0 of two differing feature dims.",
                ))
    return cases


def _buggy_add_broadcast() -> List[Case]:
    cases = []
    for in_dim in (12, 20, 40):
        for p in (6, 12, 20):
            for delta in (3, 7, -3):
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
                    id=f"{_BLIND_TAG}_buggy_add_{in_dim}_{p}_{q}",
                    family="add_broadcast_mismatch", label="buggy", domain="shape",
                    source=src, input_shapes={"x": (2, in_dim)},
                    expected_error_substring="must match the size of tensor",
                    seed_url=None, provenance_type="canonical_pattern",
                    note="[blind] Elementwise add of non-broadcastable outputs.",
                ))
    return cases


def _clean_mlp() -> List[Case]:
    cases = []
    widths_sets = [
        (96, 48, 10), (160, 80, 10), (50, 25, 10), (300, 100, 10),
        (40, 40, 40, 10), (120, 60, 30, 10),
    ]
    for in_dim in (24, 40, 96):
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
                id=f"{_BLIND_TAG}_clean_mlp_{wid}", family="clean_mlp",
                label="clean", domain="shape", source=src,
                input_shapes={"x": (8, in_dim)},
                provenance_type="canonical_clean",
                note="[blind] Idiomatic MLP with matching dimensions.",
            ))
    return cases


def _clean_conv() -> List[Case]:
    cases = []
    chan_chains = [
        (3, 12, 24), (3, 6, 12, 24), (2, 6, 12), (1, 12, 24),
    ]
    for hw in (10, 12, 20):
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
                id=f"{_BLIND_TAG}_clean_conv_{hw}_{cid}", family="clean_conv",
                label="clean", domain="shape", source=src,
                input_shapes={"x": (4, chans[0], hw, hw)},
                provenance_type="canonical_clean",
                note="[blind] Conv stack with correctly sized head.",
            ))
    return cases


def _clean_norm_mlp() -> List[Case]:
    cases = []
    for in_dim in (48, 96, 160):
        for hidden in (48, 96, 160):
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
                    id=f"{_BLIND_TAG}_clean_norm_{norm.lower()}_{in_dim}_{hidden}",
                    family="clean_norm_mlp", label="clean", domain="shape",
                    source=src, input_shapes={"x": (16, in_dim)},
                    provenance_type="canonical_clean",
                    note=f"[blind] MLP with a {norm} layer; consistent dims.",
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


def all_blind_cases() -> List[Case]:
    cases: List[Case] = []
    for gen in _GENERATORS:
        cases.extend(gen())
    cases.sort(key=lambda c: c.id)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "duplicate blind case ids"
    return cases
