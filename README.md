# tensor-index-audit

Check sharded Hugging Face checkpoints before they reach model loading. It verifies that index files point to safe, present shard files and, for SafeTensors, that the index agrees with tensor names and byte ranges in each file header.

## Problem

Large checkpoints split weights across shard files and use a JSON index to map tensor names to those shards. A download can have valid file hashes yet still have a stale or hand-edited index that points to a missing shard, maps a tensor to the wrong shard, or omits tensors. These problems otherwise surface at model startup, after a potentially expensive deployment or training job has already begun.

There is also a security boundary here: Transformers has used shard filenames from local checkpoint indexes when resolving files. A public report and an associated GitHub advisory describe path traversal through these entries. `tensor-index-audit` rejects absolute paths, parent components, and symlinks that resolve outside the checkpoint directory before it opens any shard. The report was automatically closed as stale, so this tool treats it as a concrete reason to validate paths defensively, not as a claim that every Transformers version is vulnerable.

## Solution

Run a dependency-free, read-only check in CI or just before serving a model. SafeTensors files are inspected by reading their small JSON headers only; tensor payloads are never loaded. PyTorch `.bin` shard files are checked for safe paths and existence only; the tool never unpickles or otherwise deserializes them.

## Quick Start

```sh
python -m pip install "git+https://github.com/thisbejim/tensor-index-audit.git@v0.1.0"
tensor-index-audit ./downloaded-model/
```

Use `--json` for CI or shell pipelines:

```sh
tensor-index-audit ./downloaded-model/ --json
```

Exit codes are `0` for a clean audit, `1` when inconsistencies are found, and `2` for invalid input or when no supported index is found.

## Example Output

```text
FAIL  ./downloaded-model
Indexes: 1  |  Tensors: 3  |  SafeTensors shards inspected: 2
ERROR MISSING_SHARD  model.safetensors.index.json: referenced shard 'model-00003-of-00003.safetensors' does not exist
ERROR INDEXED_TENSOR_MISSING  model.safetensors.index.json: index maps tensor 'model.layers.2.mlp.up_proj.weight' here, but it is absent from this shard
```

Try the included small fixture:

```sh
tensor-index-audit examples/tiny-checkpoint/
```

## Checks and Supported Inputs

| Input | Checks |
| --- | --- |
| `*.safetensors.index.json` | JSON structure and duplicate keys; shard paths stay within the checkpoint directory; referenced shards exist and are regular files; index tensor names match SafeTensors header names and shard locations; header lengths, descriptor shapes, fixed-width tensor sizes, payload offsets, overlap, and optional `metadata.total_size` |
| `*.bin.index.json` | JSON structure and duplicate keys; shard paths stay within the checkpoint directory; referenced shards exist and are regular files |

The checker finds supported indexes recursively, without following directory symlinks. It reads only index JSON and SafeTensors headers. It does not read tensor payloads. Default limits cap each index and header at 64 MiB; adjust them with `--max-json-bytes` and `--max-header-bytes` when needed.

An example CI step:

```yaml
- name: Audit model checkpoint
  run: tensor-index-audit ./artifacts/model/ --json
```

## Why

The Hugging Face CLI's `hf cache verify` is useful for checking local files against Hub checksums and detecting missing or extra files. It does not compare each SafeTensors index mapping with the names stored in its shards. ModelAudit and ModelScan scan a broad set of serialized model formats for malicious content; they are valuable for that job, while this small check focuses on the cross-file index contract and does not deserialize model files.

This is especially useful for teams that mirror checkpoints, assemble artifacts in CI, export models from training jobs, or mount large weights into inference workers. It catches cheap-to-detect packaging mistakes before Transformers, vLLM, or another loader starts reading model weights.

## Privacy and Security

- No network access, account, API key, or external service is required.
- No telemetry is present.
- The CLI emits relative paths and tensor names from the local checkpoint. JSON output is deterministic for a given checkpoint.
- `.bin` files are never deserialized. SafeTensors tensor payloads are not read.
- The audit is a structural preflight, not a malware detector, model correctness proof, or replacement for updating vulnerable loaders. It does not protect against a file changing between audit and use (TOCTOU), resource exhaustion outside the configured JSON/header limits, or malicious tensor values.
- The tool's path checks are defense in depth. Keep model-serving processes isolated and use patched model-loading libraries.

## Development

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
mypy
```

The tests use tiny synthetic SafeTensors headers and fixture `.bin` bytes. They need no model download, framework, GPU, API credential, or network connection.

## References

- [Hugging Face CLI: `hf cache verify`](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-cache-verify)
- [Transformers issue #47176: shard index path traversal report](https://github.com/huggingface/transformers/issues/47176)
- [GitHub advisory GHSA-fv5v-hfxp-5379 / CVE-2026-75104](https://github.com/advisories/GHSA-fv5v-hfxp-5379)
- [ModelAudit](https://github.com/promptfoo/modelaudit)
- [ModelScan](https://github.com/protectai/modelscan)

## License

MIT. See [LICENSE](LICENSE).
