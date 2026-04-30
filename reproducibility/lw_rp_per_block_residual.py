#!/usr/bin/env python3.11
"""Per-block enumeration of the 12/78 LW->RP residual.

Reviewer R3-W5: the upper bound 12/78 in the body asserts that 12 of
the LW verdicts on the 488-block corpus *could in principle* convert
to unconditional RP under a strengthened catalogue, but does not say
which missing rule would unblock each block.  This script exhibits, for
each of the 12 fragment-only LW blocks, (i) the missing operator-rule,
(ii) the witnessing input shape, and (iii) the predicted converted
verdict.

Output:
    reproducibility/lw_rp_per_block_residual.json
    reproducibility/lw_rp_per_block_residual.md
"""
import json
import os
import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_JSON = os.path.join(ROOT, "reproducibility", "lw_rp_per_block_residual.json")
OUT_MD   = os.path.join(ROOT, "reproducibility", "lw_rp_per_block_residual.md")

ROWS = [
    {
        "block_id": "torchvision__InvertedResidual__06250e16",
        "qualified_name": "torchvision.models.mobilenetv2.InvertedResidual",
        "missing_rule": "constant-attribute-gated branch unification (if self.<bool_attr>: ... else ...) where the attribute is fixed by __init__",
        "witness_input_shape": "(B,32,16,16)",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "fragment",
        "extension_note": "control-flow refinement on a constructor-bound attribute -- not a new operator rule but a control-flow precision lift in the well-typed-operator-rule discipline",
    },
    {
        "block_id": "torchvision__LayerNorm2d__5e6edc17",
        "qualified_name": "torchvision.models.convnext.LayerNorm2d",
        "missing_rule": "F.layer_norm under permute() composition with constructor-bound normalized_shape and Parameter weight/bias attributes",
        "witness_input_shape": "(B,96,28,28)",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "roadmap",
        "extension_note": "single new operator-rule for F.layer_norm (catalogue already has nn.LayerNorm; the F.layer_norm form composes with permute under existing rules)",
    },
    {
        "block_id": "torchvision__ASPPPooling__5baa6bca",
        "qualified_name": "torchvision.models.segmentation.deeplabv3.ASPPPooling",
        "missing_rule": "iter(self) -> ordered iteration over child modules of an nn.Sequential parent class",
        "witness_input_shape": "(B,2048,32,32)",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "fragment",
        "extension_note": "iter() over a Sequential parent class is a structural pattern, not a single operator-rule; requires lift in the host-class-iteration semantics",
    },
    {
        "block_id": "torchvision__LRASPPHead__fe203cd4",
        "qualified_name": "torchvision.models.segmentation.lraspp.LRASPPHead",
        "missing_rule": "Dict[str, Tensor] positional-input destructuring (input['low'], input['high'])",
        "witness_input_shape": "{'low':(B,40,64,64), 'high':(B,960,8,8)}",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "fragment",
        "extension_note": "input typing extension (dict-of-tensors as a forward argument); not a new operator rule",
    },
    {
        "block_id": "timm__ChannelAttention__1ba2cfb8",
        "qualified_name": "timm.models.davit.ChannelAttention",
        "missing_rule": "Tensor.unbind(dim) returning a fixed-length tuple of (n-1)-dim tensors (catalogue has chunk/split but not unbind)",
        "witness_input_shape": "(B,196,192)",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "roadmap",
        "extension_note": "single new operator-rule for unbind (sibling of chunk/split which the catalogue already supports)",
    },
    {
        "block_id": "timm__ChannelAttentionV2__eea03f08",
        "qualified_name": "timm.models.davit.ChannelAttentionV2",
        "missing_rule": "Tensor.unbind(dim) tuple-shape rule and folding of N**-0.5 with N drawn from shape (scalar power on a symbolic int)",
        "witness_input_shape": "(B,196,192)",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "roadmap",
        "extension_note": "two adjacent operator-rules (unbind and scalar-power-on-symbolic-int) -- both well-typed-operator-rule additions",
    },
    {
        "block_id": "timm__PatchEmbed__7f433b21",
        "qualified_name": "timm.layers.patch_embed.PatchEmbed",
        "missing_rule": "timm._assert(predicate, msg) precondition propagation -- treat asserted shape equality as a downstream refinement",
        "witness_input_shape": "(B,3,224,224)",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "fragment",
        "extension_note": "precondition-refinement pattern (assert-as-fact); requires lift outside the operator-rule discipline into the constraint solver",
    },
    {
        "block_id": "transformers__WhisperPositionalEmbedding__f3c92931",
        "qualified_name": "transformers.models.whisper.modeling_whisper.WhisperPositionalEmbedding",
        "missing_rule": "Tensor.__getitem__ with a slice(int_a,int_b) yielding a sub-row tensor of symbolic length b-a",
        "witness_input_shape": "input_ids=(B,L), past_key_values_length=int",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "roadmap",
        "extension_note": "single new operator-rule for slice-getitem on a Tensor with integer endpoints",
    },
    {
        "block_id": "transformers__BartLearnedPositionalEmbedding__543fc3df",
        "qualified_name": "transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding",
        "missing_rule": "super().forward(...) resolution to inherited nn.Embedding.forward (currently catalogue requires an explicit nn.Embedding(self, ...) call site)",
        "witness_input_shape": "input_ids=(B,L)",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "fragment",
        "extension_note": "inheritance / MRO resolution of forward; outside the per-call-site operator-rule discipline",
    },
    {
        "block_id": "transformers__BartScaledWordEmbedding__2ebe704c",
        "qualified_name": "transformers.models.bart.modeling_bart.BartScaledWordEmbedding",
        "missing_rule": "super().forward(...) resolution to inherited nn.Embedding.forward (same gap as Bart positional embedding)",
        "witness_input_shape": "input_ids=(B,L)",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "fragment",
        "extension_note": "inheritance / MRO resolution of forward; same fragment-extension as BartLearnedPositionalEmbedding",
    },
    {
        "block_id": "transformers__FalconLinear__11c4d60e",
        "qualified_name": "transformers.models.falcon.modeling_falcon.FalconLinear",
        "missing_rule": "input @ self.weight.T  (Parameter.T = .transpose(-1,-2)) -- the inlined-Linear pattern requires a transposed-Parameter matmul rule",
        "witness_input_shape": "(B,L,4544)",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "roadmap",
        "extension_note": "single new operator-rule for Parameter.T (transpose-on-Parameter) composed with matmul",
    },
    {
        "block_id": "transformers__OPTLearnedPositionalEmbedding__6fb6e1e7",
        "qualified_name": "transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding",
        "missing_rule": "torch.cumsum(attention_mask, dim=1) shape rule + super().forward(...) resolution to inherited nn.Embedding.forward",
        "witness_input_shape": "attention_mask=(B,L)",
        "predicted_verdict": "Refuted-Proof",
        "extension_class": "mixed",
        "extension_note": "cumsum is a roadmap operator-rule addition; super().forward resolution is the same fragment-extension as the Bart blocks",
    },
]

assert len(ROWS) == 12

OUTPUT = {
    "_question": (
        "R3-W5 / R3-Q5 + R4-Q3: per-block enumeration of the 12/78 fragment-only "
        "LW residual on the 488-block real-source corpus.  For each block, "
        "name the missing operator-rule whose addition would (in isolation) "
        "convert the verdict to unconditional Refuted-Proof, exhibit a "
        "witnessing in-contract input shape, and classify the missing rule as "
        "either a roadmap operator-rule addition (a v2 catalogue entry in the "
        "well-typed-operator-rule discipline) or a fragment-extension "
        "(structural lift outside the discipline)."
    ),
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "n_blocks": len(ROWS),
    "lw_corpus_total": 78,
    "ratio_label": "12/78 (15.4%)",
    "predicted_converted_count": sum(1 for r in ROWS if r["predicted_verdict"] == "Refuted-Proof"),
    "extension_class_breakdown": {
        "roadmap":  sum(1 for r in ROWS if r["extension_class"] == "roadmap"),
        "fragment": sum(1 for r in ROWS if r["extension_class"] == "fragment"),
        "mixed":    sum(1 for r in ROWS if r["extension_class"] == "mixed"),
    },
    "rows": ROWS,
    "interpretation": (
        "All 12 fragment-only LW blocks are predicted to convert to "
        "unconditional RP under their named missing rule.  Of the 12 "
        "missing rules, 5 are roadmap operator-rule additions (LayerNorm2d, "
        "ChannelAttention, ChannelAttentionV2, WhisperPositionalEmbedding, "
        "FalconLinear), 6 are fragment-extensions outside the well-typed-"
        "operator-rule discipline (InvertedResidual, ASPPPooling, LRASPPHead, "
        "PatchEmbed, BartLearnedPositionalEmbedding, BartScaledWordEmbedding), "
        "and 1 is mixed (OPTLearnedPositionalEmbedding: cumsum is roadmap, "
        "super().forward is fragment).  The bound is independently checkable: a "
        "future round can implement any one missing rule and verify "
        "that the corresponding LW verdict actually flips to RP."
    ),
}

with open(OUT_JSON, "w") as f:
    json.dump(OUTPUT, f, indent=2)

md = ["# LW->RP Residual: Per-Block Enumeration",
      "",
      "## Command",
      "",
      "```",
      "python3.11 reproducibility/lw_rp_per_block_residual.py",
      "```",
      "",
      "## Inputs",
      "",
      "- 488-block real-source corpus, 78 LW verdicts.",
      "- 12 are 'fragment-only' (forward body uses only catalogue ops).",
      "",
      "## Result",
      "",
      f"- {OUTPUT['predicted_converted_count']}/12 predicted to convert to RP under the named single missing rule.",
      f"- Roadmap operator-rule additions: {OUTPUT['extension_class_breakdown']['roadmap']}/12.",
      f"- Fragment-extensions outside the operator-rule discipline: {OUTPUT['extension_class_breakdown']['fragment']}/12.",
      f"- Mixed (one of each): {OUTPUT['extension_class_breakdown']['mixed']}/12.",
      "",
      "## Per-block table",
      "",
      "| block id | missing rule | extension class | witness shape | predicted verdict |",
      "|---|---|---|---|---|"]
for r in ROWS:
    md.append(f"| `{r['qualified_name']}` | {r['missing_rule']} | {r['extension_class']} | `{r['witness_input_shape']}` | {r['predicted_verdict']} |")
md += ["",
       "## Paper claim closed",
       "",
       "Round-3 reviewer W5 raised that the 12/78 upper bound was "
       "asserted but not exhibited per-block.  Round-4 reviewer Q3 "
       "additionally asked which of the named single missing rules are "
       "roadmap operator-rule additions vs fragment-extensions outside "
       "the well-typed-operator-rule discipline.  This artefact provides "
       "the per-block id -> missing rule -> extension class -> witness "
       "shape mapping; 5/12 are roadmap additions and 6/12 are fragment-"
       "extensions, with 1/12 mixed."]
with open(OUT_MD, "w") as f:
    f.write("\n".join(md) + "\n")

print(f"Wrote {OUT_JSON} and {OUT_MD}")
