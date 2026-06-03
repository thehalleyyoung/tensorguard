#!/usr/bin/env python3
"""Step 261 -- runtime-silent bug benchmark.

Every case in this benchmark executes real PyTorch code without raising, then an
independent semantic oracle proves the result is wrong.  TensorGuard's
object-level gates are scored separately, so the benchmark does not use the
checker as its ground-truth oracle.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "reproducibility" / "silent_bug_benchmark.json"
OUT_MD = REPO / "reproducibility" / "silent_bug_benchmark.md"


@dataclass
class CaseResult:
    case_id: str
    family: str
    runtime_nonraising: bool
    oracle_positive: bool
    gate_caught: bool
    issue_kinds: List[str]
    semantic_delta: float
    notes: str


def _seed():
    import torch
    torch.manual_seed(261)


def _step(model, x, y):
    import torch
    opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.zero_grad()
    loss = (model(x) - y).pow(2).mean()
    loss.backward()
    opt.step()
    after = {n: p.detach().clone() for n, p in model.named_parameters()}
    return before, after


def _grad_freeze_cases() -> List[Callable[[], CaseResult]]:
    import torch
    import torch.nn as nn
    from src.silent_bug_checks import verify_silent_bug_contracts
    _seed()

    class FrozenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 2)
            for p in self.fc1.parameters():
                p.requires_grad = False
            self.tensorguard_expected_trainable = ("fc1", "fc2")
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    class FrozenConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 4, 3, padding=1)
            self.head = nn.Conv2d(4, 4, 1)
            for p in self.stem.parameters():
                p.requires_grad = False
            self.tensorguard_expected_trainable = ("stem", "head")
        def forward(self, x):
            return self.head(torch.relu(self.stem(x)))

    class FrozenEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(8, 3)
            self.head = nn.Linear(3, 1)
            self.emb.weight.requires_grad = False
            self.tensorguard_expected_trainable = ("emb", "head")
        def forward(self, x):
            return self.head(self.emb(x).mean(dim=1))

    specs = [
        ("gradient_freeze_mlp", FrozenMLP, torch.randn(5, 4), torch.randn(5, 2), "fc1.weight"),
        ("gradient_freeze_conv", FrozenConv, torch.randn(2, 3, 6, 6), torch.randn(2, 4, 6, 6), "stem.weight"),
        ("gradient_freeze_embedding", FrozenEmbedding, torch.tensor([[1, 2, 3], [3, 2, 1]]), torch.randn(2, 1), "emb.weight"),
    ]

    def make(case_id, cls, x, y, frozen_name):
        def run():
            _seed()
            model = cls()
            before, after = _step(model, x, y)
            unchanged = bool(torch.equal(before[frozen_name], after[frozen_name]))
            verdict = verify_silent_bug_contracts(model)
            return CaseResult(case_id, "gradient_freeze", True, unchanged, not verdict.ok,
                              sorted({i.kind for i in verdict.issues}), 0.0,
                              f"{frozen_name} unchanged after a training step")
        return run
    return [make(*spec) for spec in specs]


def _stale_buffer_cases() -> List[Callable[[], CaseResult]]:
    import torch
    import torch.nn as nn
    from src.silent_bug_checks import verify_silent_bug_contracts
    _seed()

    class ScaleBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor([0.5, 1.0, 1.5]))
            self.tensorguard_expected_buffers = {"scale": torch.tensor([1.0, 1.0, 1.0])}
        def forward(self, x):
            return x * self.scale

    class BiasBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", torch.tensor([2.0, -2.0]))
            self.tensorguard_expected_buffers = {"bias": torch.zeros(2)}
        def forward(self, x):
            return x + self.bias

    class PosBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("pos", torch.arange(4.0).view(1, 4, 1) + 10.0)
            self.tensorguard_expected_buffers = {"pos": torch.arange(4.0).view(1, 4, 1)}
        def forward(self, x):
            return x + self.pos

    specs = [
        ("stale_buffer_scale", ScaleBuffer, torch.ones(2, 3)),
        ("stale_buffer_bias", BiasBuffer, torch.ones(3, 2)),
        ("stale_buffer_position", PosBuffer, torch.zeros(2, 4, 1)),
    ]

    def make(case_id, cls, x):
        def run():
            _seed()
            model = cls()
            out = model(x)
            expected_name, expected = next(iter(model.tensorguard_expected_buffers.items()))
            current = dict(model.named_buffers())[expected_name].detach().clone()
            getattr(model, expected_name).copy_(expected)
            ref = model(x)
            getattr(model, expected_name).copy_(current)
            delta = float((out - ref).abs().max().item())
            verdict = verify_silent_bug_contracts(model)
            return CaseResult(case_id, "stale_buffer", True, delta > 1e-6, not verdict.ok,
                              sorted({i.kind for i in verdict.issues}), round(delta, 6),
                              "forward output differs from declared-buffer reference")
        return run
    return [make(*spec) for spec in specs]


def _optimizer_drift_cases() -> List[Callable[[], CaseResult]]:
    import torch
    import torch.nn as nn
    from src.silent_bug_checks import optimizer_state_fingerprints, verify_silent_bug_contracts
    _seed()

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2, bias=False)
        def forward(self, x):
            return self.linear(x)

    def initialized():
        _seed()
        model = Tiny()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        x, y = torch.randn(4, 3), torch.randn(4, 2)
        opt.zero_grad()
        (model(x) - y).pow(2).mean().backward()
        opt.step()
        return model, opt, x, y

    def make(case_id, mutate):
        def run():
            model, opt, x, y = initialized()
            expected = optimizer_state_fingerprints(opt)
            clean_model = copy.deepcopy(model)
            clean_opt = torch.optim.AdamW(clean_model.parameters(), lr=0.01)
            clean_opt.load_state_dict(opt.state_dict())
            mutate(opt)
            before_clean = clean_model.linear.weight.detach().clone()
            before_bad = model.linear.weight.detach().clone()
            for m, o in ((clean_model, clean_opt), (model, opt)):
                o.zero_grad()
                (m(x) - y).pow(2).mean().backward()
                o.step()
            clean_delta = clean_model.linear.weight.detach() - before_clean
            bad_delta = model.linear.weight.detach() - before_bad
            delta = float((clean_delta - bad_delta).abs().max().item())
            verdict = verify_silent_bug_contracts(model, optimizer=opt, optimizer_fingerprints=expected)
            return CaseResult(case_id, "optimizer_state_drift", True, delta > 1e-9, not verdict.ok,
                              sorted({i.kind for i in verdict.issues}), round(delta, 9),
                              "same-shape AdamW state corruption changes the next update without raising")
        return run

    def zero_exp_avg(opt):
        next(iter(opt.state.values()))["exp_avg"].zero_()
    def flip_exp_avg_sq(opt):
        next(iter(opt.state.values()))["exp_avg_sq"].mul_(4.0)
    def stale_step(opt):
        next(iter(opt.state.values()))["step"].add_(100)

    return [
        make("optimizer_drift_zero_exp_avg", zero_exp_avg),
        make("optimizer_drift_scaled_exp_avg_sq", flip_exp_avg_sq),
        make("optimizer_drift_stale_step", stale_step),
    ]


def _mode_leak_cases() -> List[Callable[[], CaseResult]]:
    import torch
    import torch.nn as nn
    from src.silent_bug_checks import verify_silent_bug_contracts
    _seed()

    class DropoutTrainLeak(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(p=1.0)
            self.tensorguard_expected_training = False
        def forward(self, x):
            return self.drop(x)

    class BatchNormEvalLeak(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(3)
            self.eval()
            self.tensorguard_expected_training = True
        def forward(self, x):
            return self.bn(x)

    class NestedDropoutLeak(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(nn.Linear(3, 3), nn.Dropout(p=1.0))
            self.eval()
            self.block[1].train()
            self.tensorguard_expected_module_modes = {"block.1": False}
        def forward(self, x):
            return self.block(x)

    specs = [
        ("mode_leak_dropout_train", DropoutTrainLeak, torch.ones(2, 3)),
        ("mode_leak_batchnorm_eval", BatchNormEvalLeak, torch.randn(4, 3) + 5.0),
        ("mode_leak_nested_dropout", NestedDropoutLeak, torch.ones(2, 3)),
    ]

    def make(case_id, cls, x):
        def run():
            _seed()
            model = cls()
            out = model(x)
            ref = copy.deepcopy(model)
            if hasattr(ref, "tensorguard_expected_training"):
                ref.train(bool(ref.tensorguard_expected_training))
            for name, expected in getattr(ref, "tensorguard_expected_module_modes", {}).items():
                dict(ref.named_modules())[name].train(bool(expected))
            ref_out = ref(x)
            delta = float((out - ref_out).abs().max().item())
            verdict = verify_silent_bug_contracts(model)
            return CaseResult(case_id, "train_eval_mode_leakage", True, delta > 1e-6, not verdict.ok,
                              sorted({i.kind for i in verdict.issues}), round(delta, 6),
                              "declared train/eval contract changes forward semantics")
        return run
    return [make(*spec) for spec in specs]


def _quant_cases() -> List[Callable[[], CaseResult]]:
    import torch
    import torch.nn as nn
    from src.silent_bug_checks import verify_silent_bug_contracts
    _seed()

    class QuantRoundtrip(nn.Module):
        def __init__(self, scale, zero_point, expected_scale, expected_zero_point):
            super().__init__()
            self.scale = scale
            self.zero_point = zero_point
            self.tensorguard_quantization_contract = {
                "scale": expected_scale,
                "zero_point": expected_zero_point,
            }
        def forward(self, x):
            return torch.quantize_per_tensor(x, self.scale, self.zero_point, torch.quint8).dequantize()

    specs = [
        ("quant_wrong_scale_coarse", 1.0, 0, 0.05, 0, torch.tensor([0.04, 0.11, 0.19])),
        ("quant_wrong_zero_point", 0.1, 20, 0.1, 0, torch.tensor([-0.2, 0.1, 0.4])),
        ("quant_wrong_scale_saturation", 0.5, 0, 0.01, 0, torch.tensor([0.07, 0.13, 0.21])),
    ]

    def make(case_id, scale, zp, exp_scale, exp_zp, x):
        def run():
            model = QuantRoundtrip(scale, zp, exp_scale, exp_zp)
            out = model(x)
            ref = QuantRoundtrip(exp_scale, exp_zp, exp_scale, exp_zp)(x)
            delta = float((out - ref).abs().max().item())
            verdict = verify_silent_bug_contracts(model)
            return CaseResult(case_id, "quantization_wrong_output", True, delta > 1e-6, not verdict.ok,
                              sorted({i.kind for i in verdict.issues}), round(delta, 6),
                              "positive qparams run but dequantized output differs from declared qparams")
        return run
    return [make(*spec) for spec in specs]


def build_cases() -> List[Callable[[], CaseResult]]:
    return (
        _grad_freeze_cases()
        + _stale_buffer_cases()
        + _optimizer_drift_cases()
        + _mode_leak_cases()
        + _quant_cases()
    )


def measure() -> Dict[str, object]:
    per_case = []
    for run_case in build_cases():
        result = run_case()
        per_case.append(result.__dict__)

    by_family: Dict[str, Dict[str, int]] = {}
    for item in per_case:
        fam = item["family"]
        row = by_family.setdefault(fam, {
            "total": 0, "runtime_nonraising": 0, "oracle_positive": 0, "gate_caught": 0,
        })
        row["total"] += 1
        row["runtime_nonraising"] += int(bool(item["runtime_nonraising"]))
        row["oracle_positive"] += int(bool(item["oracle_positive"]))
        row["gate_caught"] += int(bool(item["gate_caught"]))

    total = len(per_case)
    caught = sum(1 for item in per_case if item["gate_caught"])
    return {
        "step": 261,
        "description": "runtime-silent bug benchmark with independent semantic oracles",
        "families": sorted(by_family),
        "summary": {
            "total_cases": total,
            "runtime_nonraising": sum(1 for item in per_case if item["runtime_nonraising"]),
            "oracle_positive": sum(1 for item in per_case if item["oracle_positive"]),
            "gate_caught": caught,
            "gate_recall": round(caught / total, 6) if total else 0.0,
        },
        "by_family": by_family,
        "per_case": per_case,
        "oracle_independence": (
            "Ground truth is observed by real forward/backward/optimizer execution "
            "and reference-output/update deltas; TensorGuard gates inspect declared "
            "runtime contracts separately."
        ),
        "scope_note": (
            "Curated CPU-only benchmark over author-constructed silent-failure "
            "families; it demonstrates coverage and regression protection, not a "
            "field prevalence estimate."
        ),
    }


def render_markdown(data: Dict[str, object]) -> str:
    s = data["summary"]  # type: ignore[index]
    lines = [
        "# Silent-bug benchmark (Step 261)",
        "",
        f"Curated CPU-only benchmark of **{s['total_cases']}** runtime-silent PyTorch bugs: "
        f"**{s['runtime_nonraising']}/{s['total_cases']}** execute without raising, "
        f"**{s['oracle_positive']}/{s['total_cases']}** are positive under an independent "
        f"semantic oracle, and TensorGuard object-level gates catch "
        f"**{s['gate_caught']}/{s['total_cases']}**.",
        "",
        data["scope_note"],  # type: ignore[index]
        "",
        "| family | cases | non-raising | oracle-positive | gate-caught |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for fam, row in sorted(data["by_family"].items()):  # type: ignore[union-attr]
        lines.append(
            f"| {fam} | {row['total']} | {row['runtime_nonraising']} "
            f"| {row['oracle_positive']} | {row['gate_caught']} |"
        )
    lines.extend(["", "## Cases", "", "| id | family | delta | issue kinds |", "| --- | --- | ---: | --- |"])
    for item in data["per_case"]:  # type: ignore[index]
        lines.append(
            f"| {item['case_id']} | {item['family']} | {item['semantic_delta']} "
            f"| {', '.join(item['issue_kinds'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if ok:
            print("silent_bug_benchmark: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return run(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
