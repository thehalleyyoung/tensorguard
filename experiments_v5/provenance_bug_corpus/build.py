#!/usr/bin/env python3
"""Build the provenance-rich GitHub bug corpus expansion (Step 249).

The source dataset in ``experiments_v5/github_bug_mining`` is already frozen by
hash.  This builder derives a richer, still-offline artifact from it: parsed
GitHub provenance, explicit repository-license metadata, PR/candidate commit
links, runtime signatures, and redistributable category-level reproducers.
No third-party source code is copied into the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[1]
MINED_DIR = THIS_DIR.parent / "github_bug_mining"
OUT_JSONL = THIS_DIR / "corpus.jsonl"
OUT_MANIFEST = THIS_DIR / "manifest.json"


def _load_mined_module():
    spec = importlib.util.spec_from_file_location("tg_mined_bug_load", MINED_DIR / "load.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load github_bug_mining/load.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MINED = _load_mined_module()

SCHEMA_VERSION = "tensorguard.provenance-bug-corpus.v1"
RECORD_SCHEMA_VERSION = "tensorguard.provenance-bug-record.v1"

CATEGORY_OPERATOR_FAMILY = {
    "broadcast_mismatch": "broadcast",
    "cat_stack_mismatch": "concat_stack",
    "conv_channel_mismatch": "convolution",
    "device_mismatch": "device_placement",
    "dim_out_of_range": "indexing_dimension",
    "dtype_device_input_mismatch": "dtype_device_contract",
    "matmul_linear_mismatch": "matmul_linear",
    "view_reshape_total_size": "reshape_view",
}

REPRODUCERS: Dict[str, Dict[str, Any]] = {
    "broadcast_mismatch": {
        "path": "experiments_v5/provenance_bug_corpus/reproducers/broadcast_mismatch.py",
        "expected_error_substring": "must match",
        "requires_cuda": False,
    },
    "cat_stack_mismatch": {
        "path": "experiments_v5/provenance_bug_corpus/reproducers/cat_stack_mismatch.py",
        "expected_error_substring": "Sizes of tensors must match except in dimension",
        "requires_cuda": False,
    },
    "conv_channel_mismatch": {
        "path": "experiments_v5/provenance_bug_corpus/reproducers/conv_channel_mismatch.py",
        "expected_error_substring": "Given groups=1",
        "requires_cuda": False,
    },
    "device_mismatch": {
        "path": "experiments_v5/provenance_bug_corpus/reproducers/device_mismatch.py",
        "expected_error_substring": "Expected all tensors to be on the same device",
        "requires_cuda": True,
    },
    "dim_out_of_range": {
        "path": "experiments_v5/provenance_bug_corpus/reproducers/dim_out_of_range.py",
        "expected_error_substring": "Dimension out of range",
        "requires_cuda": False,
    },
    "dtype_device_input_mismatch": {
        "path": "experiments_v5/provenance_bug_corpus/reproducers/dtype_device_input_mismatch.py",
        "expected_error_substring": "Input type",
        "requires_cuda": False,
    },
    "matmul_linear_mismatch": {
        "path": "experiments_v5/provenance_bug_corpus/reproducers/matmul_linear_mismatch.py",
        "expected_error_substring": "shapes cannot be multiplied",
        "requires_cuda": False,
    },
    "view_reshape_total_size": {
        "path": "experiments_v5/provenance_bug_corpus/reproducers/view_reshape_total_size.py",
        "expected_error_substring": "is invalid for input of size",
        "requires_cuda": False,
    },
}

# Conservative public-license snapshot for frequent repositories in the frozen
# dataset.  Long-tail repositories stay NOASSERTION rather than guessed.
LICENSE_SNAPSHOT: Dict[str, Dict[str, str]] = {
    "AUTOMATIC1111/stable-diffusion-webui": {
        "spdx_id": "AGPL-3.0",
        "evidence_url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/LICENSE.txt",
    },
    "Comfy-Org/ComfyUI": {
        "spdx_id": "GPL-3.0",
        "evidence_url": "https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE",
    },
    "VainF/Torch-Pruning": {
        "spdx_id": "MIT",
        "evidence_url": "https://github.com/VainF/Torch-Pruning/blob/master/LICENSE",
    },
    "deepspeedai/DeepSpeed": {
        "spdx_id": "Apache-2.0",
        "evidence_url": "https://github.com/deepspeedai/DeepSpeed/blob/master/LICENSE",
    },
    "huggingface/diffusers": {
        "spdx_id": "Apache-2.0",
        "evidence_url": "https://github.com/huggingface/diffusers/blob/main/LICENSE",
    },
    "huggingface/trl": {
        "spdx_id": "Apache-2.0",
        "evidence_url": "https://github.com/huggingface/trl/blob/main/LICENSE",
    },
    "kijai/ComfyUI-WanVideoWrapper": {
        "spdx_id": "Apache-2.0",
        "evidence_url": "https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/LICENSE",
    },
    "modelscope/ms-swift": {
        "spdx_id": "Apache-2.0",
        "evidence_url": "https://github.com/modelscope/ms-swift/blob/main/LICENSE",
    },
    "open-mmlab/mmdetection": {
        "spdx_id": "Apache-2.0",
        "evidence_url": "https://github.com/open-mmlab/mmdetection/blob/main/LICENSE",
    },
    "open-mmlab/mmsegmentation": {
        "spdx_id": "Apache-2.0",
        "evidence_url": "https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE",
    },
    "pyg-team/pytorch_geometric": {
        "spdx_id": "MIT",
        "evidence_url": "https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE",
    },
    "pytorch/ao": {
        "spdx_id": "BSD-3-Clause",
        "evidence_url": "https://github.com/pytorch/ao/blob/main/LICENSE",
    },
    "pytorch/executorch": {
        "spdx_id": "BSD-3-Clause",
        "evidence_url": "https://github.com/pytorch/executorch/blob/main/LICENSE",
    },
    "pytorch/pytorch": {
        "spdx_id": "BSD-3-Clause",
        "evidence_url": "https://github.com/pytorch/pytorch/blob/main/LICENSE",
    },
    "sgl-project/sglang": {
        "spdx_id": "Apache-2.0",
        "evidence_url": "https://github.com/sgl-project/sglang/blob/main/LICENSE",
    },
    "sktime/pytorch-forecasting": {
        "spdx_id": "MIT",
        "evidence_url": "https://github.com/sktime/pytorch-forecasting/blob/main/LICENSE",
    },
    "vllm-project/vllm": {
        "spdx_id": "Apache-2.0",
        "evidence_url": "https://github.com/vllm-project/vllm/blob/main/LICENSE",
    },
}

_GITHUB_RE = re.compile(r"^https://github\.com/([^/]+)/([^/]+)/(issues|pull)/(\d+)$")


def _canonical_line(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _sha256_lines(records: Iterable[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for record in records:
        h.update(_canonical_line(record).encode("utf-8"))
    return h.hexdigest()


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.write_text("".join(_canonical_line(r) for r in records), encoding="utf-8")


def _parse_github_url(url: str) -> Dict[str, Any]:
    match = _GITHUB_RE.match(url)
    if not match:
        raise ValueError(f"unsupported GitHub URL: {url}")
    owner, repo_name, kind, number = match.groups()
    return {
        "owner": owner,
        "repo_name": repo_name,
        "github_kind": "pull_request" if kind == "pull" else "issue",
        "github_number": int(number),
    }


def _license_metadata(repository: str) -> Dict[str, Any]:
    known = LICENSE_SNAPSHOT.get(repository)
    if known:
        return {
            "spdx_id": known["spdx_id"],
            "status": "known_from_static_public_snapshot",
            "evidence_url": known["evidence_url"],
            "snapshot_source": "public_repository_license_file",
            "third_party_code_redistributed": False,
        }
    return {
        "spdx_id": "NOASSERTION",
        "status": "not_cached_in_offline_snapshot",
        "evidence_url": None,
        "snapshot_source": "offline_issue_pr_metadata_only",
        "third_party_code_redistributed": False,
    }


def _project_family(record: Dict[str, Any]) -> str:
    blob = f"{record.get('repository', '')} {record.get('title', '')}".lower()
    if any(tok in blob for tok in ("comfy", "diffusion", "stable-diffusion", "sd-webui")):
        return "diffusion"
    if any(tok in blob for tok in ("yolo", "detect", "segmentation", "mmdetection")):
        return "vision_detection"
    if any(tok in blob for tok in ("llama", "qwen", "glm", "internlm", "transformer", "vllm")):
        return "language_model"
    if any(tok in blob for tok in ("graph", "geometric", "gnn")):
        return "graph_learning"
    return "unclassified_public_repo"


def _runtime_signature(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "matched_signature": record["matched_signature"],
        "domain": record["domain"],
        "category": record["category"],
        "operator_family": CATEGORY_OPERATOR_FAMILY[record["category"]],
        "source": "verbatim_pytorch_runtime_error_fragment",
    }


def _reproducer_metadata(category: str) -> Dict[str, Any]:
    info = REPRODUCERS[category]
    return {
        "scope": "category_level_minimized_reproducer",
        "path": info["path"],
        "expected_error_substring": info["expected_error_substring"],
        "requires_cuda": info["requires_cuda"],
        "legally_redistributable": True,
        "license": "MIT",
        "authors": "TensorGuard Authors",
        "third_party_code_copied": False,
    }


def _commit_index(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[str]]:
    by_repo_category: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for record in records:
        if record.get("is_pull_request"):
            key = (record["repository"], record["category"])
            by_repo_category[key].append(record["source_url"] + "/commits")
    return {k: sorted(set(v)) for k, v in by_repo_category.items()}


def _commit_links(record: Dict[str, Any], index: Dict[Tuple[str, str], List[str]]) -> Tuple[List[Dict[str, str]], str]:
    if record.get("is_pull_request"):
        return (
            [{
                "url": record["source_url"] + "/commits",
                "kind": "pull_request_commits_page",
                "status": "direct_pr_unverified_offline",
            }],
            "direct_pr_commits_page",
        )

    candidates = index.get((record["repository"], record["category"]), [])[:3]
    if candidates:
        return (
            [
                {
                    "url": url,
                    "kind": "same_repo_same_category_pr_commits_page",
                    "status": "candidate_unverified_offline",
                }
                for url in candidates
            ],
            "candidate_same_repo_category_pr",
        )
    return [], "not_available_in_offline_snapshot"


def enrich_records() -> List[Dict[str, Any]]:
    source_records = sorted(MINED.verify_integrity(), key=lambda r: (r["source_url"], r["id"]))
    commit_index = _commit_index(source_records)
    enriched: List[Dict[str, Any]] = []
    for record in source_records:
        parsed = _parse_github_url(record["source_url"])
        links, link_status = _commit_links(record, commit_index)
        enriched.append({
            "schema_version": RECORD_SCHEMA_VERSION,
            "id": record["id"],
            "source_url": record["source_url"],
            "repository": record["repository"],
            "owner": parsed["owner"],
            "repo_name": parsed["repo_name"],
            "github_kind": parsed["github_kind"],
            "github_number": parsed["github_number"],
            "title": record["title"],
            "state": record["state"],
            "created_at": record["created_at"],
            "domain": record["domain"],
            "category": record["category"],
            "project_family": _project_family(record),
            "runtime_signature": _runtime_signature(record),
            "license_metadata": _license_metadata(record["repository"]),
            "commit_links": links,
            "commit_link_status": link_status,
            "reproducer": _reproducer_metadata(record["category"]),
            "redistribution": {
                "third_party_code_redistributed": False,
                "stored_source_blob": False,
                "stored_issue_body": False,
                "stored_patch": False,
                "reproducer_is_repo_authored": True,
                "reproducer_license": "MIT",
            },
        })
    return enriched


def _count(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(r[key] for r in records).items()))


def build_manifest(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_manifest = MINED.load_manifest()["meta"]
    licenses = Counter(r["license_metadata"]["spdx_id"] for r in records)
    known_license_records = sum(
        1 for r in records
        if r["license_metadata"]["status"] == "known_from_static_public_snapshot"
    )
    known_license_repos = {
        r["repository"] for r in records
        if r["license_metadata"]["status"] == "known_from_static_public_snapshot"
    }
    link_statuses = Counter(r["commit_link_status"] for r in records)
    any_commit_link = sum(1 for r in records if r["commit_links"])
    categories = sorted(_count(records, "category"))
    missing_reproducers = [c for c in categories if c not in REPRODUCERS]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "experiments_v5/provenance_bug_corpus/build.py",
        "deterministic": True,
        "source_dataset": {
            "path": "experiments_v5/github_bug_mining/mined_bugs_dataset.jsonl",
            "manifest_path": "experiments_v5/github_bug_mining/mined_bugs_manifest.json",
            "records": source_manifest["total"],
            "sha256": source_manifest["dataset_sha256"],
        },
        "total": len(records),
        "corpus_sha256": _sha256_lines(records),
        "by_domain": _count(records, "domain"),
        "by_category": _count(records, "category"),
        "by_github_kind": _count(records, "github_kind"),
        "by_project_family": _count(records, "project_family"),
        "license_metadata": {
            "known_records": known_license_records,
            "known_repositories": len(known_license_repos),
            "noassertion_records": len(records) - known_license_records,
            "by_spdx_id": dict(sorted(licenses.items())),
            "known_snapshot_repositories": sorted(known_license_repos),
            "long_tail_policy": "NOASSERTION rather than guessed when no offline license snapshot is committed",
        },
        "commit_link_coverage": {
            "records_with_any_commit_link": any_commit_link,
            "records_without_commit_link": len(records) - any_commit_link,
            "by_status": dict(sorted(link_statuses.items())),
            "claim": (
                "PR rows carry direct PR commit-page links; issue rows carry "
                "same-repo/same-category candidate PR commit-page links when "
                "available in the frozen offline snapshot."
            ),
        },
        "runtime_signatures": {
            "source": "verbatim PyTorch runtime error fragments",
            "by_signature": dict(sorted(Counter(
                r["runtime_signature"]["matched_signature"] for r in records
            ).items())),
        },
        "reproducers": {
            "categories_covered": len(REPRODUCERS) - len(missing_reproducers),
            "missing_categories": missing_reproducers,
            "cpu_executable_categories": sorted(
                c for c, meta in REPRODUCERS.items() if not meta["requires_cuda"]
            ),
            "cuda_qualified_categories": sorted(
                c for c, meta in REPRODUCERS.items() if meta["requires_cuda"]
            ),
            "paths": {c: meta["path"] for c, meta in sorted(REPRODUCERS.items())},
            "scope": "one repo-authored minimized reproducer per runtime-signature category",
        },
        "redistribution": {
            "third_party_code_redistributed": False,
            "issue_bodies_redistributed": False,
            "patches_redistributed": False,
            "reproducer_license": "MIT",
            "legal_basis": "facts/URLs plus repo-authored minimized reproducers only",
        },
    }


def build() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = enrich_records()
    return records, build_manifest(records)


def write() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records, manifest = build()
    _write_jsonl(OUT_JSONL, records)
    OUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return records, manifest


def check() -> int:
    records, manifest = build()
    expected_jsonl = "".join(_canonical_line(r) for r in records)
    expected_manifest = json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    problems = []
    if not OUT_JSONL.exists() or OUT_JSONL.read_text(encoding="utf-8") != expected_jsonl:
        problems.append(str(OUT_JSONL.relative_to(REPO)))
    if not OUT_MANIFEST.exists() or OUT_MANIFEST.read_text(encoding="utf-8") != expected_manifest:
        problems.append(str(OUT_MANIFEST.relative_to(REPO)))
    if problems:
        print("MISMATCH: regenerate " + ", ".join(problems))
        return 1
    print(
        f"OK: {manifest['total']} provenance-rich bugs, "
        f"sha256 {manifest['corpus_sha256'][:16]} verified."
    )
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed artifacts")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    records, manifest = write()
    print(
        f"Wrote {OUT_JSONL.relative_to(REPO)} and {OUT_MANIFEST.relative_to(REPO)} "
        f"({len(records)} records, sha256 {manifest['corpus_sha256'][:16]})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
