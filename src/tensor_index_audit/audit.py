"""Read-only structural checks for Hugging Face sharded checkpoint indexes."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path, PureWindowsPath
from typing import Any

INDEX_SUFFIXES = {
    ".safetensors.index.json": "safetensors",
    ".bin.index.json": "opaque",
}
DEFAULT_MAX_JSON_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_HEADER_BYTES = 64 * 1024 * 1024
_FIXED_DTYPE_BYTES = {
    "BOOL": 1,
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "U64": 8,
    "I32": 4,
    "U32": 4,
    "I16": 2,
    "U16": 2,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
}


@dataclass(frozen=True)
class Issue:
    """One actionable finding from an index audit."""

    code: str
    severity: str
    path: str
    message: str


@dataclass
class AuditReport:
    """Serializable summary of all checkpoint indexes audited."""

    root: str
    indexes_scanned: int
    tensors_indexed: int
    safetensors_shards_read: int
    issues: list[Issue]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "root": self.root,
            "indexes_scanned": self.indexes_scanned,
            "tensors_indexed": self.tensors_indexed,
            "safetensors_shards_read": self.safetensors_shards_read,
            "issues": [asdict(issue) for issue in self.issues],
        }


class DuplicateJSONKey(ValueError):
    """Raised instead of silently accepting duplicate JSON object keys."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, limit: int) -> Any:
    size = path.stat().st_size
    if size > limit:
        raise ValueError(f"JSON file is {size} bytes; configured limit is {limit}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_object)


def _index_kind(path: Path) -> str | None:
    name = path.name
    for suffix, kind in INDEX_SUFFIXES.items():
        if name.endswith(suffix):
            return kind
    return None


def _discover(root: Path) -> list[Path]:
    found: list[Path] = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name for name in directories if not (Path(current) / name).is_symlink()
        )
        for name in files:
            candidate = Path(current) / name
            if _index_kind(candidate):
                found.append(candidate)
    return sorted(found)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_shard_path(root: Path, index_dir: Path, raw: str) -> tuple[Path | None, str | None]:
    if not raw or "\x00" in raw or "\\" in raw:
        return None, "shard filename is empty or contains a non-portable path character"
    posix = Path(raw)
    windows = PureWindowsPath(raw)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        return None, "shard filename must be relative to the checkpoint directory"
    if any(part in {"", ".", ".."} for part in raw.split("/")):
        return None, "shard filename contains an empty, dot, or parent-directory component"

    candidate = index_dir / posix
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        return None, f"cannot resolve shard path: {exc}"
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        return None, "shard path resolves outside the audited checkpoint directory"
    return candidate, None


def _issue(report: AuditReport, code: str, path: Path, root: Path, message: str) -> None:
    report.issues.append(Issue(code, "error", _relative(path, root), message))


def _tensor_descriptors(
    shard: Path,
    root: Path,
    max_header_bytes: int,
    report: AuditReport,
) -> tuple[dict[str, tuple[int, int]], int] | None:
    """Return tensor names and byte extents without reading tensor payloads."""
    try:
        file_size = shard.stat().st_size
        with shard.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("file is shorter than the 8-byte SafeTensors header prefix")
            header_size = int.from_bytes(prefix, "little", signed=False)
            if header_size == 0:
                raise ValueError("SafeTensors header is empty")
            if header_size > max_header_bytes:
                raise ValueError(
                    f"header is {header_size} bytes; configured limit is {max_header_bytes}"
                )
            data_start = 8 + header_size
            if data_start > file_size:
                raise ValueError("header length extends beyond end of file")
            raw_header = stream.read(header_size)
        header = json.loads(raw_header, object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, DuplicateJSONKey, ValueError) as exc:
        _issue(report, "INVALID_SAFETENSORS_HEADER", shard, root, str(exc))
        return None

    if not isinstance(header, dict):
        _issue(report, "INVALID_SAFETENSORS_HEADER", shard, root, "header must be a JSON object")
        return None
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        _issue(
            report,
            "INVALID_SAFETENSORS_METADATA",
            shard,
            root,
            "__metadata__ must map strings to strings",
        )
        return None

    payload_size = file_size - data_start
    descriptors: dict[str, tuple[int, int]] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            _issue(
                report,
                "INVALID_SAFETENSORS_TENSOR",
                shard,
                root,
                f"tensor entry {name!r} must have a non-empty name and object descriptor",
            )
            continue
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not dtype
            or not isinstance(shape, list)
            or len(shape) > 64
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or any(dimension > (2**63 - 1) for dimension in shape if type(dimension) is int)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            _issue(
                report,
                "INVALID_SAFETENSORS_TENSOR",
                shard,
                root,
                f"tensor {name!r} has an invalid dtype, shape, or data_offsets descriptor",
            )
            continue
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            _issue(
                report,
                "TENSOR_OFFSETS_OUT_OF_BOUNDS",
                shard,
                root,
                f"tensor {name!r} offsets [{start}, {end}] exceed payload size {payload_size}",
            )
            continue
        known_width = _FIXED_DTYPE_BYTES.get(dtype)
        if known_width is not None:
            elements = 1
            for dimension in shape:
                elements *= dimension
            expected_size = elements * known_width
            if end - start != expected_size:
                _issue(
                    report,
                    "TENSOR_SIZE_MISMATCH",
                    shard,
                    root,
                    (
                        f"tensor {name!r} has {end - start} payload bytes; "
                        f"{dtype} shape {shape} needs {expected_size}"
                    ),
                )
                continue
        descriptors[name] = (start, end)
        ranges.append((start, end, name))

    ranges.sort()
    for previous, current in pairwise(ranges):
        if current[0] < previous[1]:
            _issue(
                report,
                "OVERLAPPING_TENSOR_OFFSETS",
                shard,
                root,
                f"tensor {current[2]!r} overlaps tensor {previous[2]!r} in the data section",
            )
    return descriptors, sum(end - start for start, end in descriptors.values())


def _audit_index(
    index: Path,
    root: Path,
    kind: str,
    report: AuditReport,
    max_json_bytes: int,
    max_header_bytes: int,
) -> None:
    try:
        resolved_index = index.resolve(strict=True)
        resolved_index.relative_to(root.resolve())
        if not stat.S_ISREG(resolved_index.stat().st_mode):
            raise ValueError("index is not a regular file")
    except (OSError, RuntimeError, ValueError) as exc:
        _issue(
            report,
            "INVALID_INDEX_PATH",
            index,
            root,
            f"index is not safely inside root: {exc}",
        )
        return
    try:
        document = _read_json(index, max_json_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, DuplicateJSONKey, ValueError) as exc:
        _issue(report, "INVALID_INDEX_JSON", index, root, str(exc))
        return
    if not isinstance(document, dict):
        _issue(report, "INVALID_INDEX", index, root, "index document must be a JSON object")
        return
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        _issue(
            report, "INVALID_WEIGHT_MAP", index, root, "weight_map must be a non-empty JSON object"
        )
        return
    if any(not isinstance(name, str) or not name for name in weight_map):
        _issue(
            report,
            "INVALID_WEIGHT_NAME",
            index,
            root,
            "weight_map tensor names must be non-empty strings",
        )
        return
    if any(not isinstance(name, str) or not name for name in weight_map.values()):
        _issue(
            report,
            "INVALID_SHARD_NAME",
            index,
            root,
            "weight_map shard names must be non-empty strings",
        )
        return

    report.tensors_indexed += len(weight_map)
    shard_to_names: dict[str, set[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        shard_to_names.setdefault(shard_name, set()).add(tensor_name)

    metadata = document.get("metadata", {})
    declared_size: int | None = None
    if metadata is not None and not isinstance(metadata, dict):
        _issue(report, "INVALID_INDEX_METADATA", index, root, "metadata must be a JSON object")
    elif isinstance(metadata, dict) and "total_size" in metadata:
        value = metadata["total_size"]
        if type(value) is not int or value < 0:
            _issue(
                report,
                "INVALID_TOTAL_SIZE",
                index,
                root,
                "metadata.total_size must be a non-negative integer",
            )
        else:
            declared_size = value

    observed_size = 0
    collected: dict[str, str] = {}
    for shard_name, expected_names in sorted(shard_to_names.items()):
        shard, problem = _safe_shard_path(root, index.parent, shard_name)
        if problem:
            _issue(report, "UNSAFE_SHARD_PATH", index, root, f"{shard_name!r}: {problem}")
            continue
        assert shard is not None
        if not shard.exists():
            if shard.is_symlink():
                _issue(
                    report,
                    "UNSAFE_SHARD_PATH",
                    index,
                    root,
                    f"referenced shard {shard_name!r} is a broken or looping symlink",
                )
                continue
            _issue(
                report,
                "MISSING_SHARD",
                index,
                root,
                f"referenced shard {shard_name!r} does not exist",
            )
            continue
        try:
            mode = shard.stat().st_mode
        except OSError as exc:
            _issue(report, "UNREADABLE_SHARD", shard, root, str(exc))
            continue
        if not stat.S_ISREG(mode):
            _issue(
                report, "INVALID_SHARD_TYPE", shard, root, "referenced shard is not a regular file"
            )
            continue
        if kind != "safetensors":
            continue

        report.safetensors_shards_read += 1
        result = _tensor_descriptors(shard, root, max_header_bytes, report)
        if result is None:
            continue
        descriptors, shard_size = result
        observed_size += shard_size
        actual_names = set(descriptors)
        missing = expected_names - actual_names
        extra = actual_names - expected_names
        for tensor_name in sorted(missing):
            elsewhere = collected.get(tensor_name)
            if elsewhere:
                message = (
                    f"tensor {tensor_name!r} is stored in {elsewhere!r}, but index maps it here"
                )
                code = "TENSOR_MAPPED_TO_WRONG_SHARD"
            else:
                message = (
                    f"index maps tensor {tensor_name!r} here, but it is absent from this shard"
                )
                code = "INDEXED_TENSOR_MISSING"
            _issue(report, code, index, root, message)
        for tensor_name in sorted(extra):
            message = (
                f"shard contains tensor {tensor_name!r}, "
                "but the index does not map it to this shard"
            )
            _issue(report, "UNINDEXED_TENSOR", shard, root, message)
        for tensor_name in sorted(actual_names):
            previous = collected.get(tensor_name)
            if previous and previous != shard_name:
                _issue(
                    report,
                    "DUPLICATE_TENSOR",
                    shard,
                    root,
                    f"tensor {tensor_name!r} appears in both {previous!r} and {shard_name!r}",
                )
            else:
                collected[tensor_name] = shard_name

    if kind == "safetensors" and declared_size is not None and observed_size != declared_size:
        _issue(
            report,
            "TOTAL_SIZE_MISMATCH",
            index,
            root,
            (
                f"metadata.total_size is {declared_size}, "
                f"indexed SafeTensors payload is {observed_size} bytes"
            ),
        )


def audit_path(
    path: str | Path,
    *,
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
) -> AuditReport:
    """Audit a checkpoint directory or one supported HF checkpoint index."""
    target = Path(path)
    if max_json_bytes <= 0 or max_header_bytes <= 0:
        raise ValueError("size limits must be positive")
    try:
        resolved = target.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot access {target}: {exc}") from exc
    if not resolved.is_dir() and not resolved.is_file():
        raise ValueError(f"path is not a regular file or directory: {target}")

    if resolved.is_dir():
        root = resolved
        indexes = _discover(root)
    else:
        root = resolved.parent
        if _index_kind(resolved) is None:
            raise ValueError("file is not a supported .safetensors.index.json or .bin.index.json")
        indexes = [resolved]

    report = AuditReport(
        root=str(root), indexes_scanned=0, tensors_indexed=0, safetensors_shards_read=0, issues=[]
    )
    for index in indexes:
        kind = _index_kind(index)
        if kind is None:
            continue
        report.indexes_scanned += 1
        _audit_index(index, root, kind, report, max_json_bytes, max_header_bytes)
    return report
