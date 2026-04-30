#!/usr/bin/env python3
"""Operator-family coverage table for TensorGuard.

Introspects ``src.tensor_shapes.TORCH_SHAPE_OPS`` and
``src.tensor_shapes.NUMPY_SHAPE_OPS`` (after ``stdlib.modern_ops`` has
merged its entries in) and groups every supported operator under one of
the nine high-level *families* used in the DL4C paper:

    matmul, conv, norm, reshape, index, reduction, elementwise,
    attention, control

Each fine-grained category string from the dispatch tables (e.g.
``"linalg_eig"``, ``"sdpa"``, ``"adaptive_pool"``, ``"checkpoint"``)
is mapped to exactly one of those families by ``_FAMILY_MAP`` below.
The mapping is the single source of truth for the LaTeX table and the
bar plot in the paper.

Outputs
-------
* ``benchmarks/op_family_table.json``  -- machine-readable table.
* ``docs/paper/figs/op_families.pdf``  -- bar plot consumed by the
  paper's operator-coverage figure.

Reproduce::

    python3.11 benchmarks/op_family_table.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tensor_shapes import TORCH_SHAPE_OPS, NUMPY_SHAPE_OPS  # noqa: E402

# Mapping from the fine-grained category strings used inside
# ``tensor_shapes.py``/``modern_ops.py`` to the nine paper-level families.
# Anything not listed here defaults to "elementwise" only if it is
# explicitly the ``"elementwise"``/``"broadcast"`` category; everything
# else raises an error to keep the mapping honest.
_FAMILY_MAP = {
    # matmul-like
    "matmul": "matmul", "bmm": "matmul", "linear": "matmul",
    "addmm": "matmul", "addmv": "matmul", "addr": "matmul",
    "baddbmm": "matmul", "addbmm": "matmul",
    "einsum": "matmul", "tensordot": "matmul", "outer": "matmul",
    "inner": "matmul", "vdot": "matmul", "kron": "matmul",
    # conv-like
    "conv1d": "conv", "conv3d": "conv", "conv_transpose1d": "conv",
    "unfold": "conv", "fold": "conv", "pixel_shuffle": "conv",
    "pixel_unshuffle": "conv", "channel_shuffle": "conv",
    "interpolate": "conv", "grid_sample": "conv", "affine_grid": "conv",
    "pad": "conv",
    # normalisation / pooling / loss-like aggregators
    "norm_preserve": "norm", "adaptive_pool": "norm",
    "scalar_loss": "norm", "adaptive_log_softmax": "norm",
    # reshape / restructuring
    "reshape": "reshape", "flatten": "reshape", "squeeze": "reshape",
    "unsqueeze": "reshape", "permute": "reshape", "transpose": "reshape",
    "movedim": "reshape", "expand": "reshape", "tile": "reshape",
    "repeat_interleave": "reshape", "stack": "reshape", "cat": "reshape",
    "chunk": "reshape", "split": "reshape", "block_diag": "reshape",
    "view_as_real": "reshape", "view_as_complex": "reshape",
    # index/select/mask
    "select": "index", "narrow": "index", "gather": "index",
    "scatter": "index", "index_select": "index", "masked_select": "index",
    "nonzero": "index", "unique": "index", "bincount": "index",
    "diag": "index", "diagonal": "index", "trace": "index",
    "topk": "index", "size_tensor": "index", "numel_tensor": "index",
    # reductions
    "reduce": "reduction", "reduction": "reduction",
    "linalg_norm": "reduction", "linalg_det": "reduction",
    "linalg_slogdet": "reduction",
    # elementwise / broadcasting / creation
    "elementwise": "elementwise", "broadcast": "elementwise",
    "create": "elementwise", "like": "elementwise",
    "like_create": "elementwise", "new_tensor": "elementwise",
    "arange": "elementwise", "linspace": "elementwise",
    "meshgrid": "elementwise", "cartesian_prod": "elementwise",
    "glu": "elementwise",
    "cdist": "elementwise", "pdist": "elementwise",
    "cosine_sim": "elementwise", "pairwise_dist": "elementwise",
    "fft_c2c": "elementwise", "rfft": "elementwise", "irfft": "elementwise",
    "rfft2": "elementwise", "irfft2": "elementwise",
    "stft": "elementwise", "istft": "elementwise",
    "linalg_eig": "elementwise", "linalg_svd": "elementwise",
    "linalg_qr": "elementwise", "linalg_solve": "elementwise",
    "linalg_lstsq": "elementwise", "linalg_square": "elementwise",
    # attention pattern
    "sdpa": "attention", "mha": "attention", "rope": "attention",
    "transformer_layer": "attention", "embedding": "attention",
    "einops_rearrange": "attention", "einops_repeat": "attention",
    "einops_reduce": "attention",
    # control-flow / wrapping
    "checkpoint": "control",
    "rnn_cell": "control", "lstm_cell": "control",
}

FAMILIES = [
    "matmul", "conv", "norm", "reshape", "index",
    "reduction", "elementwise", "attention", "control",
]

# Short prose description for each family, reused in the LaTeX table caption.
FAMILY_BLURB = {
    "matmul":      "matrix products and inner-product variants",
    "conv":        "spatial convolutions and structured re-arrangers",
    "norm":        "normalisation, pooling, and loss aggregation",
    "reshape":     "shape-only re-layout (no data movement guarantees)",
    "index":       "index/scatter/gather/select operators",
    "reduction":   "axis reductions and scalar summaries",
    "elementwise": "broadcasting, creation, and pure point ops",
    "attention":   "attention/embedding/einops patterns",
    "control":     "control-flow wrappers (checkpoint, RNN/LSTM cells)",
}


def classify(category: str) -> str:
    fam = _FAMILY_MAP.get(category)
    if fam is None:
        raise KeyError(
            f"unmapped fine category {category!r}; please add it to _FAMILY_MAP"
        )
    return fam


def build() -> dict:
    rows = defaultdict(lambda: {"torch": [], "numpy": []})
    for op, cat in TORCH_SHAPE_OPS.items():
        rows[classify(cat)]["torch"].append(op)
    for op, cat in NUMPY_SHAPE_OPS.items():
        rows[classify(cat)]["numpy"].append(op)

    table = []
    for fam in FAMILIES:
        torch_ops = sorted(set(rows[fam]["torch"]))
        numpy_ops = sorted(set(rows[fam]["numpy"]))
        table.append({
            "family": fam,
            "blurb": FAMILY_BLURB[fam],
            "n_torch": len(torch_ops),
            "n_numpy": len(numpy_ops),
            "torch_ops_sample": torch_ops[:6],
            "torch_ops": torch_ops,
            "numpy_ops": numpy_ops,
        })

    summary = {
        "n_torch_total": sum(r["n_torch"] for r in table),
        "n_numpy_total": sum(r["n_numpy"] for r in table),
        "fine_category_counts_torch": dict(Counter(TORCH_SHAPE_OPS.values())),
        "fine_category_counts_numpy": dict(Counter(NUMPY_SHAPE_OPS.values())),
        "families": table,
    }
    assert summary["n_torch_total"] == len(TORCH_SHAPE_OPS), (
        summary["n_torch_total"], len(TORCH_SHAPE_OPS)
    )
    assert summary["n_numpy_total"] == len(NUMPY_SHAPE_OPS)
    return summary


def plot(summary: dict, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fams = [r["family"] for r in summary["families"]]
    n_t = [r["n_torch"] for r in summary["families"]]
    n_n = [r["n_numpy"] for r in summary["families"]]

    x = np.arange(len(fams))
    w = 0.4
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(x - w / 2, n_t, w, label="PyTorch (n=%d)" % sum(n_t),
           color="#2c7fb8")
    ax.bar(x + w / 2, n_n, w, label="NumPy (n=%d)" % sum(n_n),
           color="#f0a868")
    for i, v in enumerate(n_t):
        if v:
            ax.text(i - w / 2, v + 1, str(v), ha="center", fontsize=7)
    for i, v in enumerate(n_n):
        if v:
            ax.text(i + w / 2, v + 1, str(v), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(fams, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("# supported ops")
    ax.set_title("TensorGuard operator coverage by family", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    summary = build()
    json_out = REPO_ROOT / "benchmarks" / "op_family_table.json"
    json_out.write_text(json.dumps(summary, indent=2))
    pdf_out = REPO_ROOT / "docs" / "paper" / "figs" / "op_families.pdf"
    plot(summary, pdf_out)
    print(f"wrote {json_out}")
    print(f"wrote {pdf_out}")
    print(
        "torch=%d numpy=%d across %d families"
        % (summary["n_torch_total"], summary["n_numpy_total"], len(FAMILIES))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
