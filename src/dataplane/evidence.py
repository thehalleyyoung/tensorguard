from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def write_evidence_packet(
    path: str | os.PathLike[str],
    *,
    dataset_hash: str,
    transform_graph: Sequence[Mapping[str, object]],
    split: str,
    seed: int,
    baseline_id: str,
    metric_keys: Sequence[str],
    command: str,
    limitations: Sequence[str],
    input_paths: Iterable[str | os.PathLike[str]] = (),
    generator: str = "datarefine.write_evidence_packet",
    timestamp: str | None = None,
) -> dict:
    packet = {
        "dataset_hash": dataset_hash,
        "transform_graph": list(transform_graph),
        "split": split,
        "seed": seed,
        "baseline_id": baseline_id,
        "metric_keys": list(metric_keys),
        "command": command,
        "limitations": list(limitations),
        "_provenance": {
            "generator": generator,
            "input_hashes": _input_hashes(input_paths),
            "env_hash": _env_hash(),
            "timestamp": timestamp or _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        },
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return packet


def link_evidence_to_obligations(
    metric_obligations: Mapping[str, Sequence[str]],
    certifier_results: Mapping[str, str],
    *,
    artifact_obligations: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Connect each metric/artifact to the obligations it depends on and the
    certifier result that admitted or bounded each obligation (step 258).

    ``metric_obligations`` and ``artifact_obligations`` map a metric/artifact key to
    the obligation ids it relies on. ``certifier_results`` maps each referenced
    obligation id to a certifier outcome (e.g. ``"admitted"``, ``"empirical-required"``,
    ``"rejected"``). Every obligation cited by a metric/artifact must have a recorded
    certifier result, otherwise the link is unsupported and we refuse to fabricate one.
    """
    artifact_obligations = dict(artifact_obligations or {})
    referenced: set[str] = set()
    links: dict[str, dict[str, object]] = {}
    for bucket_name, bucket in (("metrics", metric_obligations), ("artifacts", artifact_obligations)):
        for key, obligation_ids in bucket.items():
            ids = [str(item) for item in obligation_ids]
            referenced.update(ids)
            links.setdefault(bucket_name, {})[str(key)] = {
                "obligation_ids": ids,
                "certifier_results": {oid: certifier_results[oid] for oid in ids if oid in certifier_results},
            }
    missing = sorted(oid for oid in referenced if oid not in certifier_results)
    admitted = sorted(oid for oid in referenced if certifier_results.get(oid) == "admitted")
    bounded = sorted(oid for oid in referenced if certifier_results.get(oid) not in (None, "admitted"))
    if missing:
        raise ValueError(f"obligations cited by evidence lack certifier results: {', '.join(missing)}")
    return {
        "schema_version": "datarefine.evidence_obligations.v1",
        "links": links,
        "admitted_obligations": admitted,
        "bounded_obligations": bounded,
    }


def write_refined_evidence_packet(
    path: str | os.PathLike[str],
    *,
    dataset_hash: str,
    transform_graph: Sequence[Mapping[str, object]],
    split: str,
    seed: int,
    baseline_id: str,
    metric_keys: Sequence[str],
    command: str,
    limitations: Sequence[str],
    metric_obligations: Mapping[str, Sequence[str]],
    certifier_results: Mapping[str, str],
    artifact_obligations: Mapping[str, Sequence[str]] | None = None,
    input_paths: Iterable[str | os.PathLike[str]] = (),
    generator: str = "datarefine.write_refined_evidence_packet",
    timestamp: str | None = None,
) -> dict:
    """Write an evidence packet whose metrics/artifacts carry obligation provenance."""
    obligation_links = link_evidence_to_obligations(
        metric_obligations,
        certifier_results,
        artifact_obligations=artifact_obligations,
    )
    packet = {
        "dataset_hash": dataset_hash,
        "transform_graph": list(transform_graph),
        "split": split,
        "seed": seed,
        "baseline_id": baseline_id,
        "metric_keys": list(metric_keys),
        "command": command,
        "limitations": list(limitations),
        "obligation_links": obligation_links,
        "_provenance": {
            "generator": generator,
            "input_hashes": _input_hashes(input_paths),
            "env_hash": _env_hash(),
            "timestamp": timestamp or _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        },
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return packet


def _input_hashes(input_paths: Iterable[str | os.PathLike[str]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for input_path in input_paths:
        path = Path(input_path)
        hashes[str(path)] = _sha256_file(path)
    return hashes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _env_hash() -> str:
    text = platform.platform() + "|" + platform.python_version()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
