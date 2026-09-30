"""Command line interface for tensor-index-audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tensor_index_audit.audit import DEFAULT_MAX_HEADER_BYTES, DEFAULT_MAX_JSON_BYTES, audit_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensor-index-audit",
        description="Check sharded Hugging Face checkpoint indexes without loading weights.",
    )
    parser.add_argument("path", type=Path, help="checkpoint directory or supported index JSON file")
    parser.add_argument("--json", action="store_true", help="print a stable JSON report")
    parser.add_argument(
        "--max-json-bytes",
        type=int,
        default=DEFAULT_MAX_JSON_BYTES,
        metavar="N",
        help=f"maximum bytes read from each index file (default: {DEFAULT_MAX_JSON_BYTES})",
    )
    parser.add_argument(
        "--max-header-bytes",
        type=int,
        default=DEFAULT_MAX_HEADER_BYTES,
        metavar="N",
        help=(
            f"maximum bytes read from each SafeTensors header (default: {DEFAULT_MAX_HEADER_BYTES})"
        ),
    )
    return parser


def _text_report(data: dict[str, object]) -> str:
    issues = data["issues"]
    assert isinstance(issues, list)
    lines = [
        f"{'PASS' if data['ok'] else 'FAIL'}  {data['root']}",
        (
            f"Indexes: {data['indexes_scanned']}  |  Tensors: {data['tensors_indexed']}  |  "
            f"SafeTensors shards inspected: {data['safetensors_shards_read']}"
        ),
    ]
    if not issues:
        lines.append("No index or shard inconsistencies found.")
    else:
        for issue in issues:
            assert isinstance(issue, dict)
            lines.append(
                f"{issue['severity'].upper():5} {issue['code']}  "
                f"{issue['path']}: {issue['message']}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_path(
            args.path,
            max_json_bytes=args.max_json_bytes,
            max_header_bytes=args.max_header_bytes,
        )
    except (OSError, ValueError) as exc:
        print(f"tensor-index-audit: {exc}", file=sys.stderr)
        return 2
    if report.indexes_scanned == 0:
        print("tensor-index-audit: no supported checkpoint indexes found", file=sys.stderr)
        return 2
    data = report.to_dict()
    if args.json:
        print(json.dumps(data, sort_keys=True, separators=(",", ":")))
    else:
        print(_text_report(data))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
