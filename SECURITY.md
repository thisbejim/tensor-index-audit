# Security policy

Report suspected vulnerabilities privately through the GitHub Security tab for this repository. Please include a minimal synthetic input that demonstrates the issue; do not publish private checkpoints, credentials, or sensitive paths in a public issue.

`tensor-index-audit` is a best-effort structural preflight. It is not a model malware scanner, and a clean result does not make it safe to load an untrusted model in a privileged process. Keep loaders patched and isolate model loading.
