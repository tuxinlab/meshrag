# Reporting Security Issues

The LightRAG team and community take security bugs seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

To report a security issue, please use the GitHub Security Advisory:  [Report a Vulnerability](https://github.com/tuxinlab/meshrag/security/advisories/new)

The LightRAG team will send a response indicating the next steps in handling your report. After the initial reply to your report, the security team will keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

Report security bugs in third-party modules to the person or team maintaining the module.

### Known Unresolved Advisories

The following advisories have no upstream fix. They affect only the optional
`evaluation` extra (RAG quality-assessment tooling), not the `meshrag` core or
the API server runtime — install `meshrag[evaluation]` only in trusted
environments.

| Package | Advisory | Severity | Status |
| ------- | -------- | -------- | ------ |
| `diskcache` 5.6.3 | [GHSA-w8v5-vhqr-4h9v](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v) — unsafe pickle deserialization | Medium | No fix released; pulled in via `ragas` -> `instructor` (evaluation extra only) |
| `ragas` 0.4.3 | [GHSA-95ww-475f-pr4f](https://github.com/advisories/GHSA-95ww-475f-pr4f) — SSRF in multi-modal faithfulness module | Low | No fix released; direct dependency of the evaluation extra only |

### Supported Versions

The following versions currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.2.x   | :x:                |
| 1.3.x   | :white_check_mark: |
