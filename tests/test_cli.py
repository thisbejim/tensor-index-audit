from __future__ import annotations

import json
import struct
from pathlib import Path

from tensor_index_audit.cli import main


def test_cli_json_and_error_exit(tmp_path: Path, capsys: object) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))

    exit_code = main([str(tmp_path), "--json"])
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    data = json.loads(captured.out)

    assert exit_code == 1
    assert data["ok"] is False
    assert data["issues"][0]["code"] == "MISSING_SHARD"


def test_cli_clean_text_output(tmp_path: Path, capsys: object) -> None:
    shard = tmp_path / "model.safetensors"
    header = json.dumps({"weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
    header += b" " * ((-len(header)) % 8)
    shard.write_bytes(struct.pack("<Q", len(header)) + header + b"x")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard.name}})
    )

    exit_code = main([str(tmp_path)])
    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert exit_code == 0
    assert "PASS" in captured.out
