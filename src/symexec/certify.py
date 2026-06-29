"""Command-line **safety certifier** — the product surface of the quantum leap.

TensorGuard's sound core does not merely *find* shape bugs; when it proves none
are reachable on the covered fragment it can *certify* their absence.  This CLI
exposes that capability over real files:

    python -m src.symexec.certify check  a.py b.py      # exit 0 iff all certified
    python -m src.symexec.certify emit   a.py -o a.cert # write a replayable cert
    python -m src.symexec.certify verify a.py a.cert    # re-verify offline

``check`` is the gate you put in CI: it certifies each file and exits non-zero if
*any* file could not be certified safe (a sound forced-failure bug was proven).
``emit`` writes the proof-carrying JSON certificate; ``verify`` re-derives the
verdict from the file alone, trusting nothing in the certificate but its claims.

Torch-free; standard library only.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Sequence

from .safety_certificate import (
    SafetyCertificate,
    certify_file,
    dumps_safety_certificate,
    loads_safety_certificate,
    render_safety_certificate,
    verify_certificate_file,
)
from .weights import (
    certify_weights_file,
    dumps_weights_certificate,
    loads_weights_certificate,
    render_weights_certificate,
    verify_weights_certificate,
    weights_contract_from_file,
)


def _verdict_line(path: str, cert: SafetyCertificate) -> str:
    if cert.proven_safe:
        return (
            f"  CERTIFIED  {path}  "
            f"(coverage {cert.coverage:.0%}, fp {cert.fingerprint[:12]})"
        )
    return (
        f"  NOT CERTIFIED  {path}  "
        f"({cert.sound_bug_count} sound bug(s) proven)"
    )


def _cmd_check(args: argparse.Namespace, out) -> int:
    all_safe = True
    for path in args.files:
        cert = certify_file(path)
        all_safe = all_safe and cert.proven_safe
        out.write(_verdict_line(path, cert) + "\n")
        if args.render and not cert.proven_safe:
            out.write(render_safety_certificate(cert))
    out.write(
        ("\nAll files certified safe.\n" if all_safe
         else "\nNot all files could be certified.\n")
    )
    return 0 if all_safe else 1


def _cmd_emit(args: argparse.Namespace, out) -> int:
    cert = certify_file(args.file)
    text = dumps_safety_certificate(cert)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        out.write(f"wrote certificate -> {args.output}\n")
    else:
        out.write(text + "\n")
    if args.render:
        out.write(render_safety_certificate(cert))
    # A certificate that does not prove safety is still emitted, but the command
    # signals it so `emit` can also gate.
    return 0 if cert.proven_safe else 1


def _cmd_verify(args: argparse.Namespace, out) -> int:
    with open(args.certificate, "r", encoding="utf-8") as fh:
        cert = loads_safety_certificate(fh.read())
    verification = verify_certificate_file(cert, args.file)
    for name, ok, detail in verification.checks:
        flag = "ok " if ok else "FAIL"
        out.write(f"  [{flag}] {name}: {detail}\n")
    if verification.verified:
        out.write("\nVERIFIED: the certificate holds for this file.\n")
        return 0
    out.write("\nNOT VERIFIED:\n")
    for reason in verification.reasons():
        out.write(f"  - {reason}\n")
    return 1


def _cmd_weights(args: argparse.Namespace, out) -> int:
    contract = None
    contract_partial = False
    model_contract = None
    if args.model:
        from .model_contract import derive_model_contract, model_contract_to_expected

        if not args.construct:
            out.write("error: --model requires --construct \"Model(...)\"\n")
            return 2
        with open(args.model, "r", encoding="utf-8") as fh:
            model_src = fh.read()
        model_contract = derive_model_contract(
            model_src, args.construct, filename=args.model
        )
        contract = model_contract_to_expected(model_contract)
        contract_partial = True
    elif args.expected:
        contract = weights_contract_from_file(args.expected)
    cert = certify_weights_file(
        args.file, check_finite=not args.no_finite,
        expected=contract, contract_partial=contract_partial,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(dumps_weights_certificate(cert) + "\n")
        out.write(f"wrote weights certificate -> {args.output}\n")
    if model_contract is not None:
        out.write(
            f"  model contract: {len(model_contract.params)} param(s) from "
            f"{model_contract.resolved_layers} resolved layer(s)"
            + (f", {len(model_contract.abstained)} abstention(s)"
               if model_contract.abstained else "")
            + " (partial)\n"
        )
    if cert.proven_safe:
        out.write(
            f"  CERTIFIED  {args.file}  "
            f"({cert.num_tensors} tensors, fp {cert.structural_fingerprint[:12]})\n"
        )
    else:
        out.write(
            f"  NOT CERTIFIED  {args.file}  ({len(cert.findings)} finding(s))\n"
        )
    if args.render or not cert.proven_safe:
        out.write(render_weights_certificate(cert))
    return 0 if cert.proven_safe else 1


def _cmd_weights_verify(args: argparse.Namespace, out) -> int:
    with open(args.certificate, "r", encoding="utf-8") as fh:
        cert = loads_weights_certificate(fh.read())
    contract = weights_contract_from_file(args.expected) if args.expected else None
    verification = verify_weights_certificate(cert, args.file, expected=contract)
    for name, ok, detail in verification.checks:
        flag = "ok " if ok else "FAIL"
        out.write(f"  [{flag}] {name}: {detail}\n")
    if verification.verified:
        out.write("\nVERIFIED: the weights certificate holds for this file.\n")
        return 0
    out.write("\nNOT VERIFIED:\n")
    for reason in verification.reasons():
        out.write(f"  - {reason}\n")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.symexec.certify",
        description="Certify (and re-verify) the absence of forced-failure shape "
        "bugs in PyTorch source files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser(
        "check", help="certify one or more files; exit non-zero if any is unsafe"
    )
    p_check.add_argument("files", nargs="+")
    p_check.add_argument(
        "--render", action="store_true",
        help="print the full certificate for files that are not certified",
    )
    p_check.set_defaults(func=_cmd_check)

    p_emit = sub.add_parser(
        "emit", help="write a replayable JSON safety certificate for a file"
    )
    p_emit.add_argument("file")
    p_emit.add_argument("-o", "--output", help="path to write the certificate to")
    p_emit.add_argument(
        "--render", action="store_true",
        help="also print the human-readable certificate",
    )
    p_emit.set_defaults(func=_cmd_emit)

    p_verify = sub.add_parser(
        "verify", help="re-verify a JSON certificate against a file, offline"
    )
    p_verify.add_argument("file")
    p_verify.add_argument("certificate")
    p_verify.set_defaults(func=_cmd_verify)

    p_weights = sub.add_parser(
        "weights",
        help="certify a safetensors checkpoint's weights layer (data + contract)",
    )
    p_weights.add_argument("file", help="path to a .safetensors checkpoint")
    p_weights.add_argument(
        "--expected", metavar="REF.safetensors",
        help="reference checkpoint whose name->(dtype,shape) contract must hold",
    )
    p_weights.add_argument(
        "--model", metavar="MODEL.py",
        help="derive the expected contract from model code (use with --construct)",
    )
    p_weights.add_argument(
        "--construct", metavar="EXPR",
        help="model construction, e.g. \"GPT(n_layer=12, n_embd=768)\"",
    )
    p_weights.add_argument(
        "--no-finite", action="store_true",
        help="skip the NaN/Inf scan of float tensors",
    )
    p_weights.add_argument("-o", "--output", help="write the JSON certificate here")
    p_weights.add_argument(
        "--render", action="store_true",
        help="always print the human-readable certificate",
    )
    p_weights.set_defaults(func=_cmd_weights)

    p_wverify = sub.add_parser(
        "weights-verify",
        help="re-verify a weights certificate against a checkpoint, offline",
    )
    p_wverify.add_argument("file")
    p_wverify.add_argument("certificate")
    p_wverify.add_argument(
        "--expected", metavar="REF.safetensors",
        help="reference checkpoint to fully re-verify a contract certificate",
    )
    p_wverify.set_defaults(func=_cmd_weights_verify)

    return parser


def main(argv: Sequence[str] | None = None, out=None) -> int:
    out = out if out is not None else sys.stdout
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return args.func(args, out)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
