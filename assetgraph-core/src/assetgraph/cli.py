from __future__ import annotations

import argparse
import json
from pathlib import Path

from assetgraph import __version__
from assetgraph.evidence.integrity import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assetgraph", description="AssetGraph Persistent Asset Intelligence CLI")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show core package status")
    status.add_argument("--json", action="store_true", dest="as_json")

    verify = sub.add_parser("sha256", help="Compute an evidence/source SHA-256")
    verify.add_argument("path")

    inspect = sub.add_parser("inspect-decision", help="Inspect a DecisionObject JSON envelope")
    inspect.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        payload = {
            "product": "AssetGraph Core",
            "version": __version__,
            "stage": "lawful-multidomain-intake",
            "principle": "evidence before promotion",
        }
        print(json.dumps(payload, indent=2) if args.as_json else f"AssetGraph Core {__version__} · {payload['stage']}")
        return 0
    if args.command == "sha256":
        print(sha256_file(args.path))
        return 0
    if args.command == "inspect-decision":
        payload = json.loads(Path(args.path).read_text())
        required = {"schema", "decision_object_id", "provenance", "review"}
        missing = sorted(required - set(payload))
        print(json.dumps({
            "valid_envelope": not missing,
            "missing": missing,
            "schema": payload.get("schema"),
            "decision_object_id": payload.get("decision_object_id"),
            "assets": len(payload.get("assets", payload.get("observations", []))),
            "events": len(payload.get("events", [])),
            "claims": len(payload.get("claims", [])),
        }, indent=2))
        return 0 if not missing else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
