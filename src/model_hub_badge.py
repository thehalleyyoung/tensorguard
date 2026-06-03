"""TensorGuard-verified model-hub badge and certificate bundles."""

from __future__ import annotations

import json
import pathlib
import re
import textwrap
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Mapping, Optional, Union

from src.api import AnalysisResult, verify_architecture
from src.signed_certificate import (
    build_certificate_drift_context,
    dumps_signed_certificate,
    sign_safety_certificate,
    verify_signed_certificate,
)


BUNDLE_SCHEMA = "tensorguard.model-hub-certificate-bundle.v1"
BADGE_FILENAME = "tensorguard-verified.svg"
CERTIFICATE_FILENAME = "signed_certificate.json"
MANIFEST_FILENAME = "manifest.json"
MODEL_CARD_SNIPPET_FILENAME = "model_card_snippet.md"


@dataclass(frozen=True)
class ModelHubBadgeBundle:
    """Paths and reproducibility anchors for a TensorGuard-verified badge."""

    model_id: str
    badge_markdown: str
    output_dir: pathlib.Path
    manifest_path: pathlib.Path
    certificate_path: pathlib.Path
    badge_svg_path: pathlib.Path
    model_card_snippet_path: pathlib.Path
    source_sha256: str
    manifest_sha256: str
    certificate_sha256: str
    payload_sha256: str
    proof_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "badge_markdown": self.badge_markdown,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "certificate_path": str(self.certificate_path),
            "badge_svg_path": str(self.badge_svg_path),
            "model_card_snippet_path": str(self.model_card_snippet_path),
            "source_sha256": self.source_sha256,
            "manifest_sha256": self.manifest_sha256,
            "certificate_sha256": self.certificate_sha256,
            "payload_sha256": self.payload_sha256,
            "proof_steps": self.proof_steps,
        }


def write_model_hub_badge_bundle(
    source: str,
    *,
    input_shapes: Optional[Mapping[str, tuple]] = None,
    output_dir: Union[str, pathlib.Path],
    model_id: str,
    secret: Union[str, bytes],
    filename: str = "<string>",
    issued_at: str = "1970-01-01T00:00:00+00:00",
    issuer: str = "tensorguard-model-hub",
    key_id: Optional[str] = None,
    soundness_mode: str = "sound",
    max_cegar_iterations: int = 10,
    infer_inputs: bool = True,
    include_proof: bool = False,
) -> ModelHubBadgeBundle:
    """Verify a model and write a reproducible model-hub badge bundle.

    The bundle is fail-closed: it is written only for a SAFE result whose
    embedded proof-carrying certificate locally replays and whose signed artifact
    verifies against the exact source/config/dependency/soundness drift context.
    """

    if not model_id or not model_id.strip():
        raise ValueError("model_id must not be empty")

    normalized_shapes = _normalize_input_shapes(input_shapes or {})
    config = {
        "input_shapes": {
            name: list(shape) for name, shape in sorted(normalized_shapes.items())
        },
        "soundness_mode": soundness_mode,
        "max_cegar_iterations": int(max_cegar_iterations),
        "infer_inputs": bool(infer_inputs),
        "include_proof": bool(include_proof),
    }

    result = verify_architecture(
        source,
        input_shapes=normalized_shapes,
        filename=filename,
        produce_certificates=include_proof,
        soundness_mode=soundness_mode,
        max_cegar_iterations=max_cegar_iterations,
        infer_inputs=infer_inputs,
    )
    _require_safe_replayed_result(result)

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    drift_context = build_certificate_drift_context(
        source=source,
        config=config,
        dependencies=_dependency_fingerprints(repo_root),
        soundness_contract=_read_required(repo_root / "SOUNDNESS_CONTRACT.md"),
    )
    artifact = sign_safety_certificate(
        result.safety_certificate,
        secret,
        issued_at=issued_at,
        issuer=issuer,
        key_id=key_id,
        require_proof=include_proof,
        drift_context=drift_context,
    )
    verification = verify_signed_certificate(
        artifact,
        secret,
        current_drift_context=drift_context,
        require_proof=include_proof,
        require_drift_context=True,
    )
    if not verification.ok:
        raise ValueError(f"signed certificate failed replay: {verification.reason}")

    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    certificate_path = output / CERTIFICATE_FILENAME
    badge_svg_path = output / BADGE_FILENAME
    snippet_path = output / MODEL_CARD_SNIPPET_FILENAME
    manifest_path = output / MANIFEST_FILENAME

    certificate_text = dumps_signed_certificate(artifact) + "\n"
    certificate_path.write_text(certificate_text, encoding="utf-8")
    badge_svg_path.write_text(render_badge_svg(), encoding="utf-8")

    badge_markdown = (
        f"[![TensorGuard verified](./{BADGE_FILENAME})]"
        f"(./{MANIFEST_FILENAME})"
    )
    snippet = render_model_card_snippet(
        model_id=model_id,
        badge_markdown=badge_markdown,
        manifest_path=MANIFEST_FILENAME,
        certificate_path=CERTIFICATE_FILENAME,
        payload_sha256=artifact["payload_sha256"],
        source_sha256=_hash_text(source),
        proof_steps=int(verification.proof_steps),
        command=_reproduction_command(filename, model_id, normalized_shapes, output),
    )
    snippet_path.write_text(snippet, encoding="utf-8")

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "model_id": model_id,
        "badge": {
            "label": "TensorGuard verified",
            "status": "verified",
            "markdown": badge_markdown,
            "svg": BADGE_FILENAME,
        },
        "verdict": "SAFE",
        "soundness_mode": soundness_mode,
        "source_file": filename,
        "source_sha256": _hash_text(source),
        "config": config,
        "certificate": {
            "path": CERTIFICATE_FILENAME,
            "sha256": _hash_file(certificate_path),
            "payload_sha256": artifact["payload_sha256"],
            "proof_steps": int(verification.proof_steps),
            "embedded_proof": include_proof,
            "schema": artifact["payload"]["schema"],
        },
        "model_card_snippet": {
            "path": MODEL_CARD_SNIPPET_FILENAME,
            "sha256": _hash_file(snippet_path),
        },
        "badge_svg": {
            "path": BADGE_FILENAME,
            "sha256": _hash_file(badge_svg_path),
        },
        "replay": {
            "ok": True,
            "reason": verification.reason,
            "model_name": verification.model_name,
        },
    }
    manifest_text = _canonical_json(manifest).decode("utf-8") + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    return ModelHubBadgeBundle(
        model_id=model_id,
        badge_markdown=badge_markdown,
        output_dir=output,
        manifest_path=manifest_path,
        certificate_path=certificate_path,
        badge_svg_path=badge_svg_path,
        model_card_snippet_path=snippet_path,
        source_sha256=manifest["source_sha256"],
        manifest_sha256=_hash_file(manifest_path),
        certificate_sha256=manifest["certificate"]["sha256"],
        payload_sha256=artifact["payload_sha256"],
        proof_steps=int(verification.proof_steps),
    )


def render_badge_svg() -> str:
    """Return a deterministic self-contained TensorGuard verified badge."""

    return textwrap.dedent(
        """\
        <svg xmlns="http://www.w3.org/2000/svg" width="184" height="20" role="img" aria-label="TensorGuard: verified">
          <title>TensorGuard: verified</title>
          <linearGradient id="s" x2="0" y2="100%">
            <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
            <stop offset="1" stop-opacity=".1"/>
          </linearGradient>
          <clipPath id="r"><rect width="184" height="20" rx="3" fill="#fff"/></clipPath>
          <g clip-path="url(#r)">
            <rect width="99" height="20" fill="#555"/>
            <rect x="99" width="85" height="20" fill="#2ea44f"/>
            <rect width="184" height="20" fill="url(#s)"/>
          </g>
          <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110">
            <text aria-hidden="true" x="505" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="890">TensorGuard</text>
            <text x="505" y="140" transform="scale(.1)" fill="#fff" textLength="890">TensorGuard</text>
            <text aria-hidden="true" x="1405" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="750">verified</text>
            <text x="1405" y="140" transform="scale(.1)" fill="#fff" textLength="750">verified</text>
          </g>
        </svg>
        """
    )


def render_model_card_snippet(
    *,
    model_id: str,
    badge_markdown: str,
    manifest_path: str,
    certificate_path: str,
    payload_sha256: str,
    source_sha256: str,
    proof_steps: int,
    command: str,
) -> str:
    """Render the copy-paste Markdown block for a model card."""

    return textwrap.dedent(
        f"""\
        {badge_markdown}

        ### TensorGuard verification

        `{model_id}` ships a reproducible TensorGuard certificate bundle.  The
        badge means the model's `nn.Module` architecture replayed as `SAFE` in
        sound mode with a signed TensorGuard safety certificate.

        - Manifest: `{manifest_path}`
        - Signed certificate: `{certificate_path}`
        - Certificate payload SHA-256: `{payload_sha256}`
        - Source SHA-256: `{source_sha256}`
        - Replayed proof steps: `{proof_steps}`

        Reproduce the bundle locally:

        ```bash
        {command}
        ```
        """
    )


def _require_safe_replayed_result(result: AnalysisResult) -> None:
    verdict = getattr(result, "verdict", "UNSAFE" if result.bugs else "SAFE")
    if verdict != "SAFE":
        details = "; ".join(getattr(result, "unknown_reasons", []) or [])
        if result.bugs:
            details = result.bugs[0].message
        raise ValueError(f"model is not badge-eligible: verdict={verdict}; {details}")
    if result.safety_certificate is None:
        raise ValueError("model is not badge-eligible: missing SafetyCertificate")
    replay = getattr(result, "certificate_replay", None)
    if replay is None or not replay.ok:
        reason = getattr(replay, "reason", "certificate replay did not run")
        raise ValueError(f"model is not badge-eligible: {reason}")


def _normalize_input_shapes(input_shapes: Mapping[str, tuple]) -> Dict[str, tuple]:
    normalized: Dict[str, tuple] = {}
    for name, shape in input_shapes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("input shape names must be non-empty strings")
        dims = []
        for dim in shape:
            if isinstance(dim, bool):
                raise ValueError(f"invalid boolean dimension for {name}")
            if isinstance(dim, int):
                if dim <= 0:
                    raise ValueError(f"dimension for {name} must be positive")
                dims.append(dim)
            elif isinstance(dim, str) and dim:
                dims.append(dim)
            else:
                raise ValueError(f"invalid dimension for {name}: {dim!r}")
        normalized[name] = tuple(dims)
    return normalized


def _dependency_fingerprints(repo_root: pathlib.Path) -> Dict[str, str]:
    paths = [
        repo_root / "pyproject.toml",
        repo_root / "src" / "api.py",
        repo_root / "src" / "signed_certificate.py",
        repo_root / "src" / "certificate_checker.py",
    ]
    return {str(path.relative_to(repo_root)): _hash_file(path) for path in paths}


def _read_required(path: pathlib.Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _reproduction_command(
    filename: str,
    model_id: str,
    input_shapes: Mapping[str, tuple],
    output_dir: pathlib.Path,
) -> str:
    parts = ["tensorguard", "model-hub-badge", _shell_token(filename)]
    for name, shape in sorted(input_shapes.items()):
        dims = ",".join(str(dim) for dim in shape)
        parts.extend(["-s", _shell_token(f"{name}={dims}")])
    parts.extend(["--model-id", _shell_token(model_id)])
    parts.extend(["--output", _shell_token(str(output_dir))])
    parts.extend(["--secret-env", "TENSORGUARD_CERT_SECRET"])
    return " ".join(parts)


def _shell_token(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=,+-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: pathlib.Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(data: Mapping[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
