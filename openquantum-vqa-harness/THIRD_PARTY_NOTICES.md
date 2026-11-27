# Source provenance and dependency notices

This repository is released under the [MIT License](LICENSE), copyright
2026 Ximo Wang and contributors.

## Compiler source provenance

The following compiler modules were adapted from the authors' existing
`openquantum-vqa-joint` publication harness:

| File in this repository | Original file within the publication package |
| --- | --- |
| `src/vqa_harness/compiler/optimizer.py` | `harness/optimizer.py` |
| `src/vqa_harness/compiler/unseen/optimizer.py` | `harness/unseen/optimizer.py` |
| `src/vqa_harness/compiler/sectors/optimizer.py` | `harness/sectors/optimizer.py` |

The source package was located at
`work/openquantum_joint_skill/publication/openquantum-vqa-joint` in the authors'
development workspace. These modules carry the same MIT copyright and
permission terms reproduced in this repository's `LICENSE`. The wrapper,
sampled workflow and host integration build on that compiler source.

## Runtime dependencies

Dependencies are installed separately and retain their own licenses and
notices. This repository does not relicense OpenQuantum, host applications or
third-party SDK distributions. Direct package pins are declared in
`pyproject.toml`; the standalone dependency resolution is recorded in
`uv.lock`.

| Package | Declared version | Role | Upstream |
| --- | --- | --- | --- |
| `mcp` | `1.30.0` | Stdio MCP server; required. | [Model Context Protocol Python SDK](https://github.com/modelcontextprotocol/python-sdk) |
| `qiskit` | `2.5.2` | Circuit construction and ideal/noisy state calculations; optional simulator extra. | [Qiskit](https://github.com/Qiskit/qiskit) |
| `numpy` | `2.5.3` | Arrays and seeded sampling; optional simulator extra. | [NumPy](https://github.com/numpy/numpy) |
| `scipy` | `1.18.1` | Classical ground-energy reference; optional simulator extra. | [SciPy](https://github.com/scipy/scipy) |

The default backend instead uses the versions and dependency notices of the
prepared OpenQuantum SDK. `vqa-joint doctor` reports that environment; each
completed run records the actual scientific package versions. OpenQuantum's
SDK locator and bootstrap are referenced from the user's plugin installation
and are not copied into this repository.

For dependency redistribution, retain the license files and any additional
notices shipped in the respective dependency distributions, including notices
for their transitive and bundled components.
