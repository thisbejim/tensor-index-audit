# Product thesis

For engineers packaging, mirroring, or deploying sharded Hugging Face checkpoints, `tensor-index-audit` catches broken and unsafe references between a checkpoint index and its shards before an expensive model load, because it checks SafeTensors headers directly without loading tensor payloads or importing model frameworks.

## Core workflow

```text
local checkpoint directory
        ↓
tensor-index-audit
        ↓
deterministic text/JSON report + CI exit code
```

## Demand and gap

- Transformers issue [#41174](https://github.com/vllm-project/vllm/issues/41174) documents a sharded-state load failure caused by tensor-key mismatches.
- Transformers issue [#32959](https://github.com/vllm-project/vllm/issues/32959) documents model startup failing on a missing key in a sharded checkpoint.
- Transformers issue [#47176](https://github.com/huggingface/transformers/issues/47176) and [GHSA-fv5v-hfxp-5379](https://github.com/advisories/GHSA-fv5v-hfxp-5379) describe unsafe shard paths in index metadata. The report was closed automatically as stale; the advisory lists affected and patched versions as unknown.
- Hugging Face's [`hf cache verify`](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-cache-verify) compares local files with Hub checksums and reports missing/extra files. It does not validate the logical mapping from each tensor name to the shard that actually contains it.
- ModelAudit and ModelScan scan many model serialization formats for malicious content. This project has a smaller scope: index-to-shard structural integrity. It complements those scanners and does not replace them.

The differentiator is the failure boundary: verify metadata relationships without importing PyTorch, Transformers, vLLM, or a model scanner. The result is useful offline and in artifact-build CI, including for custom or mirrored checkpoints where a Hub checksum is unavailable.

## Non-goals

- Loading or running models, tensor payloads, custom code, or pickle files.
- Verifying that weights implement the advertised architecture or are numerically correct.
- Replacing malware scanners, Hub checksum verification, or security advisories.
- Claiming that path validation makes a model loader safe or closes the reported Transformers issue.
- Downloading models or requiring any provider integration.

## Architecture and independence

The implementation uses only Python's standard library. It parses JSON indexes, validates safe relative paths and filesystem targets, then reads the 8-byte SafeTensors header-length prefix and bounded JSON header. `.bin` shards are checked only for path containment and presence. All data remains local; there is no telemetry or network code.

## Skeptical review

This may be too narrow for teams that rely on official Hub snapshots and load each model only once. The cache verifier already catches missing and corrupted downloads, and model loaders eventually validate much of the same structure when they start. The CLI is most valuable when a checkpoint is exported, mirrored, copied, or mounted repeatedly and a late loader error is costly. Its offline, no-framework behavior is a useful improvement for that exact preflight, but adoption still depends on teams adding one small CI step.
