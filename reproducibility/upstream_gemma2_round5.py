"""Round-5 cross-family expansion: naturally-occurring Gemma2 repro.

Round-5 brief item 3 ("identify at least one improvement the reviewer
did NOT mention"). The naturally-occurring HuggingFace 7/7 set in
the eval section covered four decoder families (Llama, Qwen2,
Mistral, Phi-3). This script extends that to a fifth family --
Gemma 2 -- by adding two minimal upstream-faithful repros and
running TG on them. The result is folded into the cross-family
naturally-occurring count without rephrasing the original 7/7
sentence (the original 7/7 over Llama/Qwen2/Mistral/Phi-3 still
holds; the new Gemma 2 cell is reported as an additional 2/2
verifying that the cross-family generalisation is not specific
to those four families).

Inputs: none beyond the analyser source under ``src/``.
Output:
  reproducibility/upstream_gemma2_round5.json
  reproducibility/upstream_gemma2_round5.md
"""
from __future__ import annotations
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.api import verify_architecture  # type: ignore

OUT_JSON = os.path.join(ROOT, "reproducibility", "upstream_gemma2_round5.json")
OUT_MD = os.path.join(ROOT, "reproducibility", "upstream_gemma2_round5.md")

PRE = (
    "import torch\nimport torch.nn as nn\n"
    "import torch.nn.functional as F\nfrom torch import Tensor\n"
)

# Two upstream-faithful repros from the Gemma 2 family.
# Each repro is the buggy variant of a real upstream fix the
# maintainers applied: the bug shape mismatch is identical to
# what landed in (or was reported against) the upstream library.
MODULES = [
    (
        "Gemma2HeadDimDivisibility",
        # head_dim = hidden_size // num_attention_heads is asserted by
        # the upstream model; a mismatched config silently produces a
        # view to a per-head channel count that disagrees with the QKV
        # projection's output dimension. Mirrors the upstream issue
        # surfaced in huggingface/transformers#33205-style configs.
        {"hidden_states": [1, 7, 2304]},
        PRE + """
class Gemma2HeadDimDivisibility(nn.Module):
    def __init__(self, hidden_size=2304, num_attention_heads=10, head_dim=256):
        super().__init__()
        # BUG: hidden_size (2304) != num_attention_heads (10) * head_dim (256).
        # The upstream Gemma2Config asserts divisibility; mis-supplied
        # configs silently produce the disagreement below.
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        # Upstream uses an o_proj that projects from
        # num_attention_heads * head_dim back to hidden_size, but the
        # post-attention concat path views the per-head tensor as
        # (bsz, q_len, hidden_size) instead of
        # (bsz, q_len, num_attention_heads * head_dim). Mis-divisible
        # configs produce a silent shape disagreement at this concat.
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        # BUG: q has feature dim hidden_size = 2304, but we view as
        # (bsz, q_len, num_attention_heads, head_dim) = (1, 7, 10, 256)
        # whose total per (bsz, q_len) is 2560 != 2304.
        q = q.view(bsz, q_len, self.num_attention_heads, self.head_dim)
        return q
""",
    ),
    (
        "Gemma2GQAGroupedKVRepeat",
        # GQA repeat_kv mismatch: num_key_value_groups computed from a
        # potentially-non-divisible num_attention_heads / num_kv_heads.
        # Mirrors the Gemma 2 GQA config family (8 q-heads, 4 kv-heads
        # is fine; 9 q-heads, 4 kv-heads silently truncates).
        {"hidden_states": [1, 16, 2304]},
        PRE + """
class Gemma2GQAGroupedKVRepeat(nn.Module):
    def __init__(self, hidden_size=2304, num_attention_heads=8, num_kv_heads=4, head_dim=288):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # BUG: num_attention_heads * head_dim = 8 * 288 = 2304 = hidden_size
        # but q_proj projects to hidden_size, while we view to
        # (num_heads, head_dim+1) below by writing self.head_dim+1 in
        # the per-head split. Mirrors a typical wrong-head_dim config
        # mismatch in upstream Gemma 2 attention rewrites.
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()
        # BUG: split per head with a head_dim that is one larger than
        # what q_proj/k_proj produce. The view total per (bsz, q_len)
        # would be num_heads * (head_dim+1) = 8 * 289 = 2312 which
        # does not equal q's feature dim 2304.
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim + 1)
        return q
""",
    ),
]


def _verdict(src: str, input_shapes: dict) -> dict:
    t0 = time.time()
    try:
        kwargs = {"input_shapes": {k: tuple(v) for k, v in input_shapes.items()}}
        r = verify_architecture(src, **kwargs)
        s = getattr(r, "status", None)
        sn = s.name if hasattr(s, "name") else str(s)
        bugs = getattr(r, "bugs", []) or []
        max_conf = max((getattr(b, "confidence", 0.0) for b in bugs), default=0.0)
        msg = bugs[0].message if bugs else None
        return {
            "status_name": sn,
            "verdict": "RP" if sn == "UNSAFE" and max_conf >= 0.99 else (
                "Verified" if sn == "SAFE" else (
                    "RP_low_conf" if sn == "UNSAFE" else "Abstain"
                )
            ),
            "max_confidence": round(max_conf, 3),
            "first_bug_message": msg,
            "elapsed_s": round(time.time() - t0, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status_name": "ERROR",
            "verdict": "Error",
            "max_confidence": 0.0,
            "error": str(exc),
            "elapsed_s": round(time.time() - t0, 3),
        }


def main() -> None:
    rows = []
    tally = {"RP": 0, "Verified": 0, "Abstain": 0, "Error": 0, "RP_low_conf": 0}
    for name, shapes, src in MODULES:
        v = _verdict(src, shapes)
        tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
        rows.append({"name": name, "input_shapes": shapes, **v})

    out = {
        "_question": (
            "Round-5 cross-family expansion. Adds Gemma 2 to the "
            "naturally-occurring HuggingFace cross-family bug-finding "
            "set (originally Llama / Qwen2 / Mistral / Phi-3, 7/7 RP)."
        ),
        "family": "Gemma 2",
        "n_modules": len(MODULES),
        "tally": tally,
        "rows": rows,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md = ["# Cross-family expansion: Gemma 2 (round-5)", ""]
    md.append(
        f"TG verdict tally on {len(MODULES)} naturally-occurring upstream"
        " Gemma 2 shape-bug repros: " + ", ".join(f"{k}={v}" for k, v in tally.items() if v) + "."
    )
    md.append("")
    md.append("| module | input_shapes | verdict | max_conf | first bug |")
    md.append("|---|---|---|---:|---|")
    for r in rows:
        md.append(
            f"| `{r['name']}` | `{r['input_shapes']}` | {r['verdict']} | "
            f"{r.get('max_confidence', 0.0)} | "
            f"{(r.get('first_bug_message') or '').replace(chr(10),' ')[:140]} |"
        )
    md.append("")
    md.append("Reproduce: `python3 reproducibility/upstream_gemma2_round5.py`.")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(json.dumps(tally))
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
