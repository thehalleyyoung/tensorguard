from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from .evidence import write_evidence_packet


@dataclass(frozen=True)
class EvidencePacket:
    dataset_hash: str
    transform_graph_hash: str
    split: str
    seed: int
    metric_keys: tuple[str, ...]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_hash": self.dataset_hash,
            "transform_graph_hash": self.transform_graph_hash,
            "split": self.split,
            "seed": self.seed,
            "metric_keys": list(self.metric_keys),
            "limitations": list(self.limitations),
        }


def stable_json_hash(value: Mapping[str, object] | Sequence[object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["EvidencePacket", "stable_json_hash", "write_evidence_packet"]
