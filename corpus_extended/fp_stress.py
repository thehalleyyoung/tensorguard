"""False-positive stress corpus: 100+ clean models (Step 111).

A sound verifier's single most important promise is "no false alarms". To stress
that promise we generate a large, deliberately *adversarial-for-clean* corpus:
over one hundred models that all execute correctly under eager PyTorch but
exercise the shape/broadcast/reshape/concat/normalisation corner cases most
likely to trick a shape checker into crying wolf.

Every model here is clean by construction -- ``fp_stress_validate`` (used by the
build/test path) instantiates it and runs a forward pass under real PyTorch. The
corpus is produced by parametric *families*: each family is a template swept over
a grid of widths, depths, kernel sizes, batch sizes, etc., yielding many
concrete clean models from one idiom. This gives breadth (idioms) and depth
(parameter variation) at the scale needed to make a "zero false alarms" claim
meaningful.

A "false alarm" is a ``UNSAFE`` verdict on any of these clean models. Abstention
(``UNKNOWN``) is *not* a false alarm -- a sound verifier is allowed to decline.
The scoring harness reports the false-alarm count (target: zero) separately from
the abstention count (a coverage metric).

Each entry is ``StressModel(id, family, source, input_shapes)``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StressModel:
    id: str
    family: str
    source: str
    input_shapes: dict


_IMPORTS = ("import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n")


def _wrap(body: str) -> str:
    return _IMPORTS + "\n\n" + body


_MODELS: list = []


def _add(id_: str, family: str, input_shapes: dict, body: str) -> None:
    _MODELS.append(StressModel(id_, family, _wrap(body), input_shapes))


# --- Family 1: deep MLPs of varied width/depth --------------------------------
def _gen_mlp():
    for width in (16, 32, 64, 128, 256):
        for depth in (1, 2, 3, 4, 5):
            init = [f"        self.l{i} = nn.Linear({width}, {width})"
                    for i in range(depth)]
            fwd = [f"        x = F.relu(self.l{i}(x))" for i in range(depth)]
            body = (
                "class Net(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                + "\n".join(init) + "\n"
                "    def forward(self, x):\n"
                + "\n".join(fwd) + "\n"
                "        return x\n"
            )
            _add(f"mlp_w{width}_d{depth}", "mlp_sweep", {"x": (4, width)}, body)


# --- Family 2: conv stacks with residual + BN (varied channels) --------------
def _gen_conv():
    for ch in (8, 16, 32, 64):
        for k in (3, 5, 7):
            pad = k // 2
            body = (
                "class Net(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                f"        self.c1 = nn.Conv2d({ch}, {ch}, {k}, padding={pad})\n"
                f"        self.c2 = nn.Conv2d({ch}, {ch}, {k}, padding={pad})\n"
                f"        self.bn = nn.BatchNorm2d({ch})\n"
                "    def forward(self, x):\n"
                "        h = F.relu(self.bn(self.c1(x)))\n"
                "        return F.relu(x + self.c2(h))\n"
            )
            _add(f"conv_ch{ch}_k{k}", "conv_res_sweep",
                 {"x": (2, ch, 16, 16)}, body)


# --- Family 3: broadcasting arithmetic (clean but tricky) --------------------
def _gen_broadcast():
    for width in (8, 16, 32, 64, 96, 128):
        body = (
            "class Net(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.f = nn.Linear({width}, {width})\n"
            "    def forward(self, x):\n"
            "        h = self.f(x)\n"
            "        bias = x[:, :1]\n"
            "        scale = h.mean(dim=-1, keepdim=True)\n"
            "        return h * scale + bias\n"
        )
        _add(f"broadcast_w{width}", "broadcast_sweep", {"x": (4, width)}, body)


# --- Family 4: reshape / view / flatten round-trips --------------------------
def _gen_reshape():
    for (c, h, w) in ((4, 4, 4), (8, 2, 2), (3, 8, 8), (16, 2, 2),
                      (6, 4, 4), (12, 2, 4), (2, 8, 8), (32, 2, 2)):
        flat = c * h * w
        body = (
            "class Net(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.f = nn.Linear({flat}, {flat})\n"
            "    def forward(self, x):\n"
            "        b = x.shape[0]\n"
            "        y = x.view(b, -1)\n"
            "        y = self.f(y)\n"
            f"        return y.view(b, {c}, {h}, {w})\n"
        )
        _add(f"reshape_{c}_{h}_{w}", "reshape_sweep",
             {"x": (2, c, h, w)}, body)


# --- Family 5: concat of parallel branches -----------------------------------
def _gen_concat():
    for nb in (2, 3, 4, 5):
        for out in (4, 8, 12):
            branches = [f"        self.b{j} = nn.Linear(16, {out})"
                        for j in range(nb)]
            cats = ", ".join(f"self.b{j}(x)" for j in range(nb))
            total = nb * out
            body = (
                "class Net(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                + "\n".join(branches) + "\n"
                f"        self.head = nn.Linear({total}, 2)\n"
                "    def forward(self, x):\n"
                f"        y = torch.cat([{cats}], dim=-1)\n"
                "        return self.head(y)\n"
            )
            _add(f"concat_n{nb}_o{out}", "concat_sweep", {"x": (4, 16)}, body)


# --- Family 6: normalisation variants ----------------------------------------
def _gen_norm():
    for width in (32, 64, 96, 128, 256):
        body = (
            "class Net(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.ln = nn.LayerNorm({width})\n"
            f"        self.f1 = nn.Linear({width}, {width * 2})\n"
            f"        self.f2 = nn.Linear({width * 2}, {width})\n"
            "    def forward(self, x):\n"
            "        h = self.ln(x)\n"
            "        return x + self.f2(F.gelu(self.f1(h)))\n"
        )
        _add(f"norm_w{width}", "norm_sweep", {"x": (8, 12, width)}, body)
    for groups in (1, 2, 4, 8, 16):
        ch = 16
        body = (
            "class Net(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.c = nn.Conv2d({ch}, {ch}, 3, padding=1)\n"
            f"        self.gn = nn.GroupNorm({groups}, {ch})\n"
            "    def forward(self, x):\n"
            "        return F.relu(self.gn(self.c(x)))\n"
        )
        _add(f"groupnorm_g{groups}", "norm_sweep", {"x": (2, ch, 8, 8)}, body)


# --- Family 7: pooling / downsampling ----------------------------------------
def _gen_pool():
    for stride in (1, 2):
        for ch in (8, 16, 32, 64):
            body = (
                "class Net(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                f"        self.c = nn.Conv2d(3, {ch}, 3, stride={stride}, "
                "padding=1)\n"
                "        self.pool = nn.AdaptiveAvgPool2d(1)\n"
                f"        self.fc = nn.Linear({ch}, 5)\n"
                "    def forward(self, x):\n"
                "        x = F.relu(self.c(x))\n"
                "        x = self.pool(x)\n"
                "        return self.fc(torch.flatten(x, 1))\n"
            )
            _add(f"pool_s{stride}_ch{ch}", "pool_sweep",
                 {"x": (2, 3, 16, 16)}, body)


# --- Family 8: attention-flavoured (matmul + softmax) ------------------------
def _gen_attn():
    for dim in (16, 32, 48, 64, 96, 128, 192):
        body = (
            "class Net(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.q = nn.Linear({dim}, {dim})\n"
            f"        self.k = nn.Linear({dim}, {dim})\n"
            f"        self.v = nn.Linear({dim}, {dim})\n"
            "    def forward(self, x):\n"
            "        q, k, v = self.q(x), self.k(x), self.v(x)\n"
            f"        a = torch.softmax(q @ k.transpose(-2, -1) "
            f"/ {float(dim) ** 0.5}, dim=-1)\n"
            "        return a @ v\n"
        )
        _add(f"attn_d{dim}", "attn_sweep", {"x": (2, 10, dim)}, body)


# --- Family 9: skip/residual chains of varied length -------------------------
def _gen_residual():
    for n in (2, 3, 4, 5, 6, 7, 8):
        width = 32
        blocks = []
        for i in range(n):
            blocks.append(f"        self.a{i} = nn.Linear({width}, {width})")
            blocks.append(f"        self.b{i} = nn.Linear({width}, {width})")
        fwd = []
        for i in range(n):
            fwd.append(f"        x = x + self.b{i}(F.relu(self.a{i}(x)))")
        body = (
            "class Net(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            + "\n".join(blocks) + "\n"
            "    def forward(self, x):\n"
            + "\n".join(fwd) + "\n"
            "        return x\n"
        )
        _add(f"residual_n{n}", "residual_sweep", {"x": (4, width)}, body)


# --- Family 10: Sequential containers of varied size -------------------------
def _gen_sequential():
    for hidden in (32, 64, 96, 128, 192, 256):
        body = (
            "class Net(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.net = nn.Sequential(\n"
            f"            nn.Linear(50, {hidden}), nn.ReLU(),\n"
            f"            nn.Linear({hidden}, {hidden}), nn.ReLU(),\n"
            f"            nn.Linear({hidden}, 10))\n"
            "    def forward(self, x):\n"
            "        return self.net(x)\n"
        )
        _add(f"seq_h{hidden}", "sequential_sweep", {"x": (8, 50)}, body)


_GENERATORS = [
    _gen_mlp, _gen_conv, _gen_broadcast, _gen_reshape, _gen_concat,
    _gen_norm, _gen_pool, _gen_attn, _gen_residual, _gen_sequential,
]

for _g in _GENERATORS:
    _g()


def all_models() -> list:
    return list(_MODELS)
