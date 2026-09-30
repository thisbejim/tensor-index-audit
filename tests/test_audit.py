from __future__ import annotations

import ctypes
import json
import struct
from pathlib import Path

import pytest
import safetensors

from tensor_index_audit import audit_path


def write_safetensors(
    path: Path, tensors: dict[str, dict[str, object]], payload: bytes = b"\0\0\0\0"
) -> None:
    header = json.dumps(tensors, separators=(",", ":")).encode()
    padding = (-len(header)) % 8
    header += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def write_index(
    directory: Path, weight_map: dict[str, str], *, total_size: int | None = None
) -> Path:
    metadata = {} if total_size is None else {"total_size": total_size}
    index = directory / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": metadata, "weight_map": weight_map}), encoding="utf-8")
    return index


def test_valid_sharded_checkpoint_is_audited_without_loading_payload(tmp_path: Path) -> None:
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    write_safetensors(
        first,
        {"embed.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        b"data",
    )
    write_safetensors(
        second,
        {"lm_head.weight": {"dtype": "I16", "shape": [1], "data_offsets": [0, 2]}},
        b"xy",
    )
    write_index(
        tmp_path,
        {
            "embed.weight": first.name,
            "lm_head.weight": second.name,
        },
        total_size=6,
    )

    report = audit_path(tmp_path)

    assert report.ok
    assert report.indexes_scanned == 1
    assert report.tensors_indexed == 2
    assert report.safetensors_shards_read == 2
    assert report.issues == []


def test_official_safetensors_writer_output_is_accepted(tmp_path: Path) -> None:
    payload = ctypes.create_string_buffer(b"\0\0\x80?", 4)
    spec = safetensors.TensorSpec(
        dtype="float32", shape=[1], data_ptr=ctypes.addressof(payload), data_len=4
    )
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(safetensors.serialize({"demo.weight": spec}))
    write_index(tmp_path, {"demo.weight": shard.name}, total_size=4)

    report = audit_path(tmp_path)

    assert report.ok
    assert report.safetensors_shards_read == 1


def test_missing_and_unindexed_tensors_are_reported(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    write_safetensors(
        shard,
        {
            "actual.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "extra.weight": {"dtype": "F32", "shape": [0], "data_offsets": [4, 4]},
        },
    )
    write_index(tmp_path, {"expected.weight": shard.name}, total_size=4)

    report = audit_path(tmp_path)

    assert {issue.code for issue in report.issues} == {
        "INDEXED_TENSOR_MISSING",
        "UNINDEXED_TENSOR",
    }


@pytest.mark.parametrize(
    "shard_name",
    [
        "../outside.safetensors",
        "/tmp/outside.safetensors",
        "C:\\outside.safetensors",
        "./model.safetensors",
        "nested//model.safetensors",
    ],
)
def test_unsafe_shard_paths_are_rejected(tmp_path: Path, shard_name: str) -> None:
    write_index(tmp_path, {"weight": shard_name})

    report = audit_path(tmp_path)

    assert not report.ok
    assert "UNSAFE_SHARD_PATH" in {issue.code for issue in report.issues}


def test_symlink_that_resolves_outside_root_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.safetensors"
    outside.write_bytes(b"not read")
    link = tmp_path / "outside-link.safetensors"
    link.symlink_to(outside)
    write_index(tmp_path, {"weight": link.name})

    report = audit_path(tmp_path)

    assert "UNSAFE_SHARD_PATH" in {issue.code for issue in report.issues}


def test_symlink_loop_is_reported_instead_of_crashing(tmp_path: Path) -> None:
    link = tmp_path / "loop.safetensors"
    link.symlink_to(link)
    write_index(tmp_path, {"weight": link.name})

    report = audit_path(tmp_path)

    assert "UNSAFE_SHARD_PATH" in {issue.code for issue in report.issues}


def test_tensor_offsets_must_not_overlap(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    write_safetensors(
        shard,
        {
            "first": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "second": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
        },
        payload=b"12345678",
    )
    write_index(tmp_path, {"first": shard.name, "second": shard.name})

    report = audit_path(tmp_path)

    assert "OVERLAPPING_TENSOR_OFFSETS" in {issue.code for issue in report.issues}


def test_duplicate_json_keys_are_not_silently_overwritten(tmp_path: Path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text('{"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}')

    report = audit_path(tmp_path)

    assert "INVALID_INDEX_JSON" in {issue.code for issue in report.issues}
    assert "duplicate JSON key" in report.issues[0].message


def test_safetensors_header_bounds_are_checked(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", 100) + b"{}")
    write_index(tmp_path, {"weight": shard.name})

    report = audit_path(tmp_path)

    assert "INVALID_SAFETENSORS_HEADER" in {issue.code for issue in report.issues}


def test_bin_index_checks_paths_without_deserializing_pickle(tmp_path: Path) -> None:
    (tmp_path / "pytorch_model-00001-of-00001.bin").write_bytes(b"not a pickle and never opened")
    index = tmp_path / "pytorch_model.bin.index.json"
    index.write_text(
        json.dumps({"weight_map": {"weight": "pytorch_model-00001-of-00001.bin"}}),
        encoding="utf-8",
    )

    report = audit_path(tmp_path)

    assert report.ok
    assert report.safetensors_shards_read == 0


def test_total_size_mismatch_is_reported(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    write_safetensors(
        shard,
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
    )
    write_index(tmp_path, {"weight": shard.name}, total_size=3)

    report = audit_path(tmp_path)

    assert "TOTAL_SIZE_MISMATCH" in {issue.code for issue in report.issues}


def test_file_input_and_size_limits(tmp_path: Path) -> None:
    index = write_index(tmp_path, {"weight": "missing.safetensors"})

    assert audit_path(index).indexes_scanned == 1
    with pytest.raises(ValueError, match="size limits must be positive"):
        audit_path(tmp_path, max_json_bytes=0)
