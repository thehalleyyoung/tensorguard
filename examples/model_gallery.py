"""Step 276 -- copyable TensorGuard model gallery.

The gallery intentionally uses small, dependency-free ``nn.Module`` programs:
every entry is real PyTorch code a user can paste into a file, every clean
variant executes under eager PyTorch, and every paired bug is checked by
TensorGuard rather than described only in prose.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "model_gallery.json")
MD_PATH = os.path.join(HERE, "model_gallery.md")


@dataclass(frozen=True)
class GalleryCase:
    slug: str
    title: str
    family: str
    input_shapes: Dict[str, Tuple[object, ...]]
    clean_source: str
    buggy_source: str
    clean_class: str
    buggy_class: str
    caught_bug: str
    copy_config: str

    @property
    def filename(self) -> str:
        return f"{self.slug}.py"


_CASE_ROWS: Tuple[Tuple[str, str, str, int, int, int, int], ...] = (
    ("mlp_classifier_head", "MLP classifier head", "tabular", 10, 20, 5, 30),
    ("residual_projection_head", "Residual projection head", "vision", 12, 24, 6, 18),
    ("transformer_ffn_block", "Transformer feed-forward block", "attention", 16, 32, 8, 40),
    ("bert_pooler_head", "BERT-style pooler head", "nlp", 768, 64, 2, 96),
    ("vit_patch_classifier", "ViT patch classifier", "vision", 64, 128, 10, 96),
    ("unet_time_mlp", "U-Net timestep MLP", "diffusion", 32, 64, 32, 48),
    ("gan_discriminator_head", "GAN discriminator head", "generative", 100, 50, 1, 40),
    ("vae_latent_decoder", "VAE latent decoder stem", "generative", 20, 40, 64, 32),
    ("recommender_dense_tower", "Recommender dense tower", "recommender", 14, 28, 1, 21),
    ("speech_ctc_projection", "Speech CTC projection", "speech", 80, 160, 29, 128),
    ("rl_policy_head", "RL policy head", "reinforcement", 17, 34, 6, 24),
    ("q_value_estimator", "Q-value estimator", "reinforcement", 22, 44, 8, 33),
    ("metric_learning_embedder", "Metric-learning embedder", "retrieval", 48, 96, 16, 72),
    ("siamese_projection_head", "Siamese projection head", "retrieval", 36, 72, 18, 54),
    ("contrastive_text_head", "Contrastive text head", "multimodal", 128, 256, 64, 192),
    ("image_caption_bridge", "Image-caption bridge", "multimodal", 256, 128, 512, 96),
    ("tabnet_decision_head", "TabNet decision head", "tabular", 30, 60, 2, 45),
    ("graph_node_classifier", "Graph node classifier", "graph", 42, 84, 7, 63),
    ("time_series_forecaster", "Time-series forecaster", "forecasting", 24, 48, 12, 36),
    ("anomaly_detector_head", "Anomaly detector head", "monitoring", 18, 36, 1, 27),
    ("autofix_regression_head", "Regression head", "tabular", 11, 22, 3, 17),
    ("adapter_bottleneck", "Adapter bottleneck", "fine-tuning", 40, 10, 40, 12),
    ("lora_merge_probe", "LoRA merge probe", "fine-tuning", 64, 16, 64, 20),
    ("optimizer_resume_probe", "Optimizer resume probe", "training", 9, 18, 9, 13),
    ("serving_schema_head", "Serving schema head", "serving", 15, 30, 4, 20),
)


def _class_name(slug: str, suffix: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_")) + suffix


def _source(class_name: str, in_features: int, hidden: int, out_features: int, second_in: int) -> str:
    return textwrap.dedent(
        f"""
        import torch.nn as nn

        class {class_name}(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear({in_features}, {hidden}),
                    nn.ReLU(),
                    nn.Linear({second_in}, {out_features}),
                )

            def forward(self, x):
                return self.net(x)
        """
    ).strip() + "\n"


def gallery_cases() -> Tuple[GalleryCase, ...]:
    cases: List[GalleryCase] = []
    for slug, title, family, in_features, hidden, out_features, bad_second in _CASE_ROWS:
        clean_class = _class_name(slug, "Clean")
        buggy_class = _class_name(slug, "Bug")
        input_shapes = {"x": ("batch", in_features)}
        cases.append(
            GalleryCase(
                slug=slug,
                title=title,
                family=family,
                input_shapes=input_shapes,
                clean_source=_source(clean_class, in_features, hidden, out_features, hidden),
                buggy_source=_source(buggy_class, in_features, hidden, out_features, bad_second),
                clean_class=clean_class,
                buggy_class=buggy_class,
                caught_bug=(
                    f"Second Linear expects {bad_second} features, but the "
                    f"previous layer produces {hidden}."
                ),
                copy_config=f"tensorguard verify {slug}.py -s x=batch,{in_features}",
            )
        )
    return tuple(cases)


def manifest() -> Dict[str, object]:
    cases = gallery_cases()
    return {
        "schema": "tensorguard.model_gallery.v1",
        "case_count": len(cases),
        "families": sorted({case.family for case in cases}),
        "cases": [
            {
                "slug": case.slug,
                "title": case.title,
                "family": case.family,
                "filename": case.filename,
                "clean_class": case.clean_class,
                "buggy_class": case.buggy_class,
                "input_shapes": {k: list(v) for k, v in case.input_shapes.items()},
                "copy_config": case.copy_config,
                "caught_bug": case.caught_bug,
                "clean_source": case.clean_source,
                "buggy_source": case.buggy_source,
            }
            for case in cases
        ],
    }


def _markdown_rows(cases: Iterable[GalleryCase]) -> List[str]:
    rows = [
        "# TensorGuard model gallery",
        "",
        "Twenty-five pasteable PyTorch `nn.Module` cases. Each row includes a clean verification path, a paired caught bug, and the minimal command/config users can copy.",
        "",
        "| # | Model | Family | Caught bug | Copy command |",
        "|---:|---|---|---|---|",
    ]
    for idx, case in enumerate(cases, 1):
        rows.append(
            f"| {idx} | `{case.slug}` | {case.family} | {case.caught_bug} | `{case.copy_config}` |"
        )
    rows.extend(
        [
            "",
            "## Example snippet",
            "",
            "Each manifest row contains `clean_source` and `buggy_source`. The clean source executes in eager PyTorch; the buggy source is expected to return `UNSAFE` under TensorGuard.",
            "",
        ]
    )
    return rows


def markdown() -> str:
    return "\n".join(_markdown_rows(gallery_cases())) + "\n"


def write_artifacts(json_path: str = JSON_PATH, md_path: str = MD_PATH) -> Tuple[str, str]:
    payload = manifest()
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(markdown())
    return json_path, md_path


def run(check: bool = False) -> int:
    if check:
        expected_json = json.dumps(manifest(), indent=2, sort_keys=True) + "\n"
        with open(JSON_PATH, "r", encoding="utf-8") as fh:
            if fh.read() != expected_json:
                print("examples/model_gallery.json is stale; run `python examples/model_gallery.py`")
                return 1
        with open(MD_PATH, "r", encoding="utf-8") as fh:
            if fh.read() != markdown():
                print("examples/model_gallery.md is stale; run `python examples/model_gallery.py`")
                return 1
        print("model gallery artifacts up to date")
        return 0
    write_artifacts()
    print(f"model gallery artifacts written: {len(gallery_cases())} cases")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="assert committed artifacts are fresh")
    args = parser.parse_args()
    return run(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
