"""Real-bug injection benchmark for TensorGuard.

For each (real public-repo file, hand-crafted shape bug):
  - apply the diff in memory to produce a buggy variant
  - run TensorGuard verify_architecture on it
  - if dynamically importable, also run torch.fx ShapeProp and FakeTensorMode
    in isolated subprocesses (60s timeout each)
  - record the verdict, first-message, first-line, and localization quality

Outputs benchmarks/injected_bugs.json and benchmarks/injected_bugs_table.tex.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch

torch.manual_seed(0)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture  # noqa: E402

CACHE_REAL = ROOT / "experiments" / ".cache" / "real_repo"
CACHE_NEW = ROOT / "benchmarks" / ".cache" / "injected_bugs"
RUNNER = ROOT / "benchmarks" / "_injected_bugs_runner.py"
OUT_JSON = ROOT / "benchmarks" / "injected_bugs.json"
OUT_TEX = ROOT / "benchmarks" / "injected_bugs_table.tex"

PYBIN = "/opt/homebrew/bin/python3.11"
SUBPROCESS_TIMEOUT = 60


# --- Source provenance (URL + commit SHA captured at benchmark construction) ---
SOURCES = {
    "nanoGPT_model.py": {
        "path": CACHE_REAL / "nanoGPT_model.py",
        "url": "https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py",
        "commit_sha": "3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
        "importable": True,
    },
    "minGPT_model.py": {
        "path": CACHE_REAL / "minGPT_model.py",
        "url": "https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/model.py",
        "commit_sha": "37baab71b9abea1b76ab957409a1cc2fbfba8a26",
        "importable": False,  # requires `mingpt` package
    },
    "mae_models.py": {
        "path": CACHE_REAL / "mae_models.py",
        "url": "https://raw.githubusercontent.com/facebookresearch/mae/main/models_mae.py",
        "commit_sha": "efb2a8062c206524e35e47d04501ed4f544c0ae8",
        "importable": False,  # requires `timm` and project-local util
    },
    "unet_parts.py": {
        "path": CACHE_REAL / "unet_parts.py",
        "url": "https://raw.githubusercontent.com/milesial/Pytorch-UNet/master/unet/unet_parts.py",
        "commit_sha": "21d7850f2af30a9695bbeea75f3136aa538cfc4a",
        "importable": True,
    },
    "labml_mha.py": {
        "path": CACHE_REAL / "labml_mha.py",
        "url": "https://raw.githubusercontent.com/labmlai/annotated_deep_learning_paper_implementations/master/labml_nn/transformers/mha.py",
        "commit_sha": "33ab02281c2b928e6b32792909cc79cbdcfe1d6a",
        "importable": False,  # requires `labml`
    },
    "resnet_basic_block.py": {
        "path": CACHE_NEW / "resnet_basic_block.py",
        "url": "https://raw.githubusercontent.com/pytorch/vision/v0.16.0/torchvision/models/resnet.py",
        "commit_sha": "fbb4cc54ed521ba912f50f180dc16a213775bf5c",
        "note": "BasicBlock + conv3x3/conv1x1 helpers extracted into a self-contained file.",
        "importable": True,
    },
    "vit_encoder_block.py": {
        "path": CACHE_NEW / "vit_encoder_block.py",
        "url": "https://raw.githubusercontent.com/pytorch/vision/v0.16.0/torchvision/models/vision_transformer.py",
        "commit_sha": "fbb4cc54ed521ba912f50f180dc16a213775bf5c",
        "note": "EncoderBlock + a minimal MLPBlock extracted into a self-contained file.",
        "importable": True,
    },
    "mlp_mixer_block.py": {
        "path": CACHE_NEW / "mlp_mixer_block.py",
        "url": "https://raw.githubusercontent.com/lucidrains/mlp-mixer-pytorch/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py",
        "commit_sha": "b102a9b25cb8bd2360db36cbd252dc4e670fd5e4",
        "note": "Single MixerBlock distilled to a self-contained einops-free nn.Module.",
        "importable": True,
    },
}


# --- Bug catalogue. Each bug specifies the exact (line, original_text, replaced_text). ---
# Lines are 1-indexed. The diff is applied iff source_lines[line-1] == original_text exactly.

BUGS = {
    "nanoGPT_model.py": [
        {
            "id": "nanogpt_split_axis",
            "description": "Split QKV along the wrong axis (dim=1 instead of dim=2).",
            "line": 56,
            "original": "        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)",
            "replaced": "        q, k, v  = self.c_attn(x).split(self.n_embd, dim=1)",
        },
        {
            "id": "nanogpt_head_dim_off_by_one",
            "description": "Wrong head_dim: C // n_head + 1 instead of C // n_head.",
            "line": 57,
            "original": "        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)",
            "replaced": "        k = k.view(B, T, self.n_head, C // self.n_head + 1).transpose(1, 2) # (B, nh, T, hs)",
        },
        {
            "id": "nanogpt_swapped_view",
            "description": "Swapped reshape dims on attention output: view(B,C,T) instead of view(B,T,C).",
            "line": 72,
            "original": "        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side",
            "replaced": "        y = y.transpose(1, 2).contiguous().view(B, C, T) # re-assemble all head outputs side by side",
        },
    ],
    "minGPT_model.py": [
        {
            "id": "mingpt_split_axis",
            "description": "Split QKV along the wrong axis (dim=1 instead of dim=2).",
            "line": 56,
            "original": "        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)",
            "replaced": "        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=1)",
        },
        {
            "id": "mingpt_head_dim_off_by_one",
            "description": "Wrong head_dim in k.view: C // n_head + 1.",
            "line": 57,
            "original": "        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)",
            "replaced": "        k = k.view(B, T, self.n_head, C // self.n_head + 1).transpose(1, 2) # (B, nh, T, hs)",
        },
        {
            "id": "mingpt_swapped_view",
            "description": "Swapped reshape dims on attention output: view(B,C,T) instead of view(B,T,C).",
            "line": 67,
            "original": "        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side",
            "replaced": "        y = y.transpose(1, 2).contiguous().view(B, C, T) # re-assemble all head outputs side by side",
        },
    ],
    "mae_models.py": [
        {
            "id": "mae_decoder_embed_swap",
            "description": "Linear in/out swap on decoder_embed projection.",
            "line": 47,
            "original": "        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)",
            "replaced": "        self.decoder_embed = nn.Linear(decoder_embed_dim, embed_dim, bias=True)",
        },
        {
            "id": "mae_decoder_pred_swap",
            "description": "Linear in/out swap on the decoder_pred head.",
            "line": 58,
            "original": "        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch",
            "replaced": "        self.decoder_pred = nn.Linear(patch_size**2 * in_chans, decoder_embed_dim, bias=True) # decoder to patch",
        },
        {
            "id": "mae_unpatchify_swapped_dims",
            "description": "Swapped (channels, batch) in unpatchify reshape.",
            "line": 120,
            "original": "        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))",
            "replaced": "        imgs = x.reshape(shape=(3, x.shape[0], h * p, h * p))",
        },
    ],
    "unet_parts.py": [
        {
            "id": "unet_conv_padding_off_by_one",
            "description": "Off-by-one in conv padding: padding=0 instead of padding=1 breaks DoubleConv shape.",
            "line": 16,
            "original": "            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),",
            "replaced": "            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0, bias=False),",
        },
        {
            "id": "unet_up_mid_channels_wrong",
            "description": "Wrong mid_channels for the bilinear-Up DoubleConv (out_channels//2 vs in_channels//2).",
            "line": 51,
            "original": "            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)",
            "replaced": "            self.conv = DoubleConv(in_channels, out_channels, out_channels // 2)",
        },
        {
            "id": "unet_outconv_in_out_swap",
            "description": "Swapped in/out channels on the OutConv 1x1 projection.",
            "line": 74,
            "original": "        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)",
            "replaced": "        self.conv = nn.Conv2d(out_channels, in_channels, kernel_size=1)",
        },
    ],
    "labml_mha.py": [
        {
            "id": "labml_prepare_linear_swap",
            "description": "Linear in/out swap on PrepareForMultiHeadAttention projection.",
            "line": 47,
            "original": "        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)",
            "replaced": "        self.linear = nn.Linear(heads * d_k, d_model, bias=bias)",
        },
        {
            "id": "labml_swapped_view_dims",
            "description": "Swapped (heads, d_k) in the view that splits the projection.",
            "line": 63,
            "original": "        x = x.view(*head_shape, self.heads, self.d_k)",
            "replaced": "        x = x.view(*head_shape, self.d_k, self.heads)",
        },
        {
            "id": "labml_d_k_off_by_one",
            "description": "Off-by-one head_dim: d_model // heads + 1 instead of d_model // heads.",
            "line": 99,
            "original": "        self.d_k = d_model // heads",
            "replaced": "        self.d_k = d_model // heads + 1",
        },
    ],
    "resnet_basic_block.py": [
        {
            "id": "resnet_conv3x3_padding_off_by_one",
            "description": "Off-by-one in conv3x3 padding: padding=0 instead of padding=dilation.",
            "line": 10,
            "original": "                     padding=dilation, groups=groups, bias=False, dilation=dilation)",
            "replaced": "                     padding=0, groups=groups, bias=False, dilation=dilation)",
        },
        {
            "id": "resnet_conv2_in_out_swap",
            "description": "Wrong channels on conv2: planes -> 2*planes (channel mismatch with bn2/identity).",
            "line": 28,
            "original": "        self.conv2 = conv3x3(planes, planes)",
            "replaced": "        self.conv2 = conv3x3(planes, planes * 2)",
        },
        {
            "id": "resnet_bn2_wrong_features",
            "description": "BatchNorm2d with wrong num_features (planes*2 instead of planes).",
            "line": 29,
            "original": "        self.bn2 = norm_layer(planes)",
            "replaced": "        self.bn2 = norm_layer(planes * 2)",
        },
    ],
    "vit_encoder_block.py": [
        {
            "id": "vit_mlp_linear1_swap",
            "description": "Linear in/out swap on MLPBlock.linear_1.",
            "line": 13,
            "original": "        self.linear_1 = nn.Linear(in_dim, mlp_dim)",
            "replaced": "        self.linear_1 = nn.Linear(mlp_dim, in_dim)",
        },
        {
            "id": "vit_mlp_linear2_dim_off",
            "description": "Off-by-one on MLPBlock.linear_2 input dim (mlp_dim+1).",
            "line": 16,
            "original": "        self.linear_2 = nn.Linear(mlp_dim, in_dim)",
            "replaced": "        self.linear_2 = nn.Linear(mlp_dim + 1, in_dim)",
        },
        {
            "id": "vit_mha_wrong_embed_dim",
            "description": "MultiheadAttention embed_dim set to mlp_dim instead of hidden_dim.",
            "line": 44,
            "original": "        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)",
            "replaced": "        self.self_attention = nn.MultiheadAttention(mlp_dim, num_heads, dropout=attention_dropout, batch_first=True)",
        },
    ],
    "mlp_mixer_block.py": [
        {
            "id": "mixer_token_mix_wrong_dim",
            "description": "token_mix expects num_patches but is built with dim, producing a shape mismatch on the transposed input.",
            "line": 27,
            "original": "        self.token_mix = FeedForward(num_patches, token_hidden, dropout)",
            "replaced": "        self.token_mix = FeedForward(dim, token_hidden, dropout)",
        },
        {
            "id": "mixer_channel_mix_wrong_dim",
            "description": "channel_mix built with num_patches instead of dim.",
            "line": 29,
            "original": "        self.channel_mix = FeedForward(dim, channel_hidden, dropout)",
            "replaced": "        self.channel_mix = FeedForward(num_patches, channel_hidden, dropout)",
        },
        {
            "id": "mixer_swapped_transpose",
            "description": "Missing transpose-back: output of token_mix is fed in token-shape rather than patch-shape.",
            "line": 34,
            "original": "        y = self.token_mix(y).transpose(1, 2)",
            "replaced": "        y = self.token_mix(y)",
        },
    ],
}


# --- How to run dynamic tools (FX / FakeTensor) on each importable file ---
RUNTIME_SPECS = {
    "nanoGPT_model.py": {
        "target_class": "CausalSelfAttention",
        "ctor_kwargs": {
            "config": {
                "__kind__": "tiny_gpt_config",
                "fields": {
                    "n_embd": 32,
                    "n_head": 4,
                    "block_size": 16,
                    "dropout": 0.0,
                    "bias": True,
                },
            }
        },
        "input_shape": [1, 16, 32],
        "input_dtype": "float",
    },
    "unet_parts.py": {
        "target_class": "DoubleConv",
        "ctor_kwargs": {"in_channels": 3, "out_channels": 16},
        "input_shape": [1, 3, 64, 64],
        "input_dtype": "float",
    },
    "resnet_basic_block.py": {
        "target_class": "BasicBlock",
        "ctor_kwargs": {"inplanes": 16, "planes": 16},
        "input_shape": [1, 16, 32, 32],
        "input_dtype": "float",
    },
    "vit_encoder_block.py": {
        "target_class": "EncoderBlock",
        "ctor_kwargs": {
            "num_heads": 4,
            "hidden_dim": 32,
            "mlp_dim": 64,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        "input_shape": [1, 16, 32],
        "input_dtype": "float",
    },
    "mlp_mixer_block.py": {
        "target_class": "MixerBlock",
        "ctor_kwargs": {
            "num_patches": 16,
            "dim": 32,
            "token_hidden": 64,
            "channel_hidden": 64,
            "dropout": 0.0,
        },
        "input_shape": [1, 16, 32],
        "input_dtype": "float",
    },
}


# --- Input shapes for TG (file-level convention) ---
TG_INPUT_SHAPES = {
    "nanoGPT_model.py": {"x": (1, 16)},
    "minGPT_model.py": {"x": (1, 16)},
    "mae_models.py": {"x": (1, 3, 224, 224)},
    "unet_parts.py": {"x": (1, 3, 256, 256)},
    "labml_mha.py": {"query": (16, 1, 64), "key": (16, 1, 64), "value": (16, 1, 64)},
    "resnet_basic_block.py": {"x": (1, 16, 32, 32)},
    "vit_encoder_block.py": {"input": (1, 16, 32)},
    "mlp_mixer_block.py": {"x": (1, 16, 32)},
}


def apply_diff(source: str, line: int, original: str, replaced: str) -> str:
    lines = source.split("\n")
    idx = line - 1
    if idx >= len(lines):
        raise ValueError(f"line {line} out of range ({len(lines)} lines)")
    if lines[idx] != original:
        raise ValueError(
            f"line {line} mismatch.\n  expected: {original!r}\n  actual:   {lines[idx]!r}"
        )
    lines[idx] = replaced
    return "\n".join(lines)


def run_tensorguard(source: str, input_shapes: dict, filename: str) -> dict:
    t0 = time.perf_counter()
    try:
        result = verify_architecture(source, input_shapes=input_shapes, filename=filename)
    except Exception as e:
        return {
            "verdict": "tool-error",
            "first_message": f"{type(e).__name__}: {e}"[:300],
            "first_line": None,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            "n_bugs": 0,
            "abstained": False,
        }
    bugs = list(getattr(result, "bugs", []) or [])
    abstained = bool(getattr(result, "abstained", False))
    shape_bugs = [
        b for b in bugs
        if b.severity == "error" and (
            "SHAPE" in b.message.upper()
            or "MODEL_CHECK" in b.message.upper()
            or "TYPE" in b.message.upper()
        )
    ]
    if shape_bugs:
        verdict = "detected"
        b = shape_bugs[0]
        first_msg = b.message[:300]
        first_line = getattr(b.location, "line", 0) or None
    elif abstained and not bugs:
        verdict = "abstain"
        first_msg = "abstained (opaque layers)"
        first_line = None
    elif bugs:
        # warnings only — don't count as a confident detection
        verdict = "abstain"
        first_msg = bugs[0].message[:300]
        first_line = getattr(bugs[0].location, "line", 0) or None
    else:
        verdict = "missed"
        first_msg = ""
        first_line = None
    return {
        "verdict": verdict,
        "first_message": first_msg,
        "first_line": first_line,
        "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
        "n_bugs": len(bugs),
        "abstained": abstained,
    }


def run_subprocess_tool(buggy_source: str, file_key: str, tool: str, bug_id: str) -> dict:
    rt = RUNTIME_SPECS.get(file_key)
    if rt is None:
        return {"verdict": "import-failed", "first_message": "no runtime spec (dependency missing)", "duration_ms": 0.0}
    payload = {
        "source": buggy_source,
        "target_class": rt["target_class"],
        "ctor_kwargs": rt["ctor_kwargs"],
        "input_shape": rt["input_shape"],
        "input_dtype": rt["input_dtype"],
        "tool": tool,
        "module_name": f"{file_key.replace('.py','')}__{bug_id}__{tool}",
    }
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [PYBIN, str(RUNNER)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"verdict": "tool-error", "first_message": "subprocess timeout", "duration_ms": SUBPROCESS_TIMEOUT * 1000.0}
    dt = round((time.perf_counter() - t0) * 1000, 2)
    out = (proc.stdout or "").strip()
    try:
        data = json.loads(out)
    except Exception:
        return {
            "verdict": "tool-error",
            "first_message": f"runner non-json: rc={proc.returncode} stderr={proc.stderr[:300]}",
            "duration_ms": dt,
        }
    return {
        "verdict": data.get("verdict", "tool-error"),
        "first_message": data.get("message", "")[:300],
        "duration_ms": dt,
    }


def main() -> None:
    records: list[dict] = []
    for fname, meta in SOURCES.items():
        path: Path = meta["path"]
        if not path.exists():
            print(f"[skip] {fname}: source not cached at {path}")
            continue
        base_source = path.read_text()
        for bug in BUGS[fname]:
            try:
                buggy = apply_diff(base_source, bug["line"], bug["original"], bug["replaced"])
            except Exception as e:
                print(f"[diff-fail] {fname}/{bug['id']}: {e}")
                continue

            print(f"[run] {fname} :: {bug['id']}  (line {bug['line']})")
            tg = run_tensorguard(buggy, TG_INPUT_SHAPES[fname], filename=fname)

            # Run dynamic tools only if file is importable
            if meta.get("importable"):
                fx = run_subprocess_tool(buggy, fname, "fx", bug["id"])
                ft = run_subprocess_tool(buggy, fname, "fakettensor", bug["id"])
            else:
                fx = {"verdict": "import-failed", "first_message": "static dependency missing", "duration_ms": 0.0}
                ft = {"verdict": "import-failed", "first_message": "static dependency missing", "duration_ms": 0.0}

            tg_loc = None
            if tg["verdict"] == "detected" and tg["first_line"]:
                tg_loc = abs(int(tg["first_line"]) - int(bug["line"])) <= 5

            rec = {
                "source_file": fname,
                "source_url": meta["url"],
                "commit_sha": meta["commit_sha"],
                "bug_id": bug["id"],
                "bug_description": bug["description"],
                "diff_line": bug["line"],
                "original_text": bug["original"],
                "replaced_text": bug["replaced"],
                "tg_verdict": tg["verdict"],
                "tg_first_message": tg["first_message"],
                "tg_first_line": tg["first_line"],
                "tg_localization_correct": tg_loc,
                "tg_duration_ms": tg["duration_ms"],
                "fx_verdict": fx["verdict"],
                "fx_first_message": fx["first_message"],
                "fx_duration_ms": fx["duration_ms"],
                "fakettensor_verdict": ft["verdict"],
                "fakettensor_first_message": ft["first_message"],
                "fakettensor_duration_ms": ft["duration_ms"],
            }
            records.append(rec)
            print(f"    tg={tg['verdict']:9s}  fx={fx['verdict']:13s}  ft={ft['verdict']:13s}")

    # Aggregate summary
    def count(field: str, value: str) -> int:
        return sum(1 for r in records if r[field] == value)

    tg_detected = count("tg_verdict", "detected")
    tg_abstain = count("tg_verdict", "abstain")
    tg_missed = count("tg_verdict", "missed")
    tg_error = count("tg_verdict", "tool-error")
    fx_detected = count("fx_verdict", "detected")
    fx_missed = count("fx_verdict", "missed")
    fx_error = count("fx_verdict", "tool-error") + count("fx_verdict", "import-failed")
    ft_detected = count("fakettensor_verdict", "detected")
    ft_missed = count("fakettensor_verdict", "missed")
    ft_error = count("fakettensor_verdict", "tool-error") + count("fakettensor_verdict", "import-failed")

    tg_loc_correct = sum(1 for r in records if r["tg_localization_correct"] is True)
    tg_loc_total = sum(1 for r in records if r["tg_localization_correct"] is not None)
    tg_loc_acc = (tg_loc_correct / tg_loc_total) if tg_loc_total else None

    summary = {
        "total_bugs": len(records),
        "n_source_files": len({r["source_file"] for r in records}),
        "tensorguard": {
            "detected": tg_detected,
            "abstain": tg_abstain,
            "missed": tg_missed,
            "tool_error": tg_error,
            "localization_correct_within_5_lines": tg_loc_correct,
            "localization_evaluated": tg_loc_total,
            "localization_accuracy": tg_loc_acc,
        },
        "torch_fx_shapeprop": {
            "detected": fx_detected,
            "missed": fx_missed,
            "error_or_import_failed": fx_error,
        },
        "fake_tensor_mode": {
            "detected": ft_detected,
            "missed": ft_missed,
            "error_or_import_failed": ft_error,
        },
    }

    md = (
        "| Tool | Detected | Abstain | Missed | Error/Unsupported |\n"
        "|---|---|---|---|---|\n"
        f"| TensorGuard | {tg_detected} | {tg_abstain} | {tg_missed} | {tg_error} |\n"
        f"| torch.fx ShapeProp | {fx_detected} | - | {fx_missed} | {fx_error} |\n"
        f"| FakeTensorMode | {ft_detected} | - | {ft_missed} | {ft_error} |\n"
    )

    OUT_JSON.write_text(json.dumps({
        "summary_markdown": md,
        "summary": summary,
        "records": records,
    }, indent=2, default=str))

    n = max(len(records), 1)
    tex = (
        "% Auto-generated by benchmarks/injected_bugs.py — do not edit by hand.\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Tool & Detected & Missed & Abstain/Error \\\\\n"
        "\\midrule\n"
        f"TensorGuard & {tg_detected}/{n} ({tg_detected/n*100:.0f}\\%)"
        f" & {tg_missed}/{n} ({tg_missed/n*100:.0f}\\%)"
        f" & {tg_abstain + tg_error}/{n} ({(tg_abstain+tg_error)/n*100:.0f}\\%) \\\\\n"
        f"torch.fx ShapeProp & {fx_detected}/{n} ({fx_detected/n*100:.0f}\\%)"
        f" & {fx_missed}/{n} ({fx_missed/n*100:.0f}\\%)"
        f" & {fx_error}/{n} ({fx_error/n*100:.0f}\\%) \\\\\n"
        f"FakeTensorMode & {ft_detected}/{n} ({ft_detected/n*100:.0f}\\%)"
        f" & {ft_missed}/{n} ({ft_missed/n*100:.0f}\\%)"
        f" & {ft_error}/{n} ({ft_error/n*100:.0f}\\%) \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    OUT_TEX.write_text(tex)

    print()
    print(md)
    print(f"[done] wrote {OUT_JSON}")
    print(f"[done] wrote {OUT_TEX}")
    print(f"[done] total bugs={len(records)}  TG det/abs/miss = {tg_detected}/{tg_abstain}/{tg_missed}")
    print(f"[done] FX detected = {fx_detected}    FakeTensor detected = {ft_detected}")
    if tg_loc_acc is not None:
        print(f"[done] TG localization accuracy: {tg_loc_correct}/{tg_loc_total} = {tg_loc_acc:.2%}")


if __name__ == "__main__":
    main()
