# OpenQuantum VQA Harness

A local, reproducible compiler and execution harness for declared variational
quantum jobs. Give it a parameterized preparation, Pauli Hamiltonian, device
graph, native gates, physical timings, model fidelities and parameter grid. It
returns a compiled execution strategy, native programs, a sampled answer and
the costs of the complete job.

The current workflow is finite-grid VQE: preparation, native lowering and
routing, Pauli-basis measurements, sampled training, parameter selection and a
fresh terminal measurement batch. Both the original and rewritten programs use
the same lowering and execution model. The compiler combines backward Pauli
pruning and affine-sector compression, then independently checks ideal Pauli
expectations at every supplied parameter binding.

## Certified joint compilation

The Python function `compile_joint(task, platform, tolerance=...)`, CLI `vqa-joint compile-joint task.json --platform NAME --out receipt.json --tolerance 0`, and MCP tool `compile_joint_job` return the same certified policy, lower and upper bounds, gap and scope. A zero gap yields `optimal`. At the search limit, a positive gap within the specified absolute tolerance yields `epsilon_optimal`. An `open` result has an empty delivery field and stores any feasible diagnostic programme under `incumbent`. The caller's tolerance overrides the platform default. Values use the registered objective unit. The CLI and MCP load trusted named adapters from the process environment; for example `VQA_JOINT_ADAPTERS='{"haituo-four-spin":"certify:HaituoFourSpin"}'` with the study's Haituo adapter on the Python path. The existing `run` CLI and `compile_vqa_job` MCP tool retain the finite-grid execution workflow with a sampled answer.

## Install and run

Requires Python 3.12 or later. The commands below use `uv` and the committed
lockfile. The default backend also requires Node.js and an installed
OpenQuantum plugin with its `hamiltonian-simulation` SDK already prepared
through OpenQuantum's normal setup.

From this checkout:

```sh
uv sync --frozen --no-editable
.venv/bin/vqa-joint doctor
.venv/bin/vqa-joint run examples/three-spin-vqe.json --out runs/demo --backend openquantum
```

The OpenQuantum backend discovers `sdk-access.mjs` in the host's plugin cache,
asks it to inspect `hamiltonian-simulation`, and runs the worker in that
prepared SDK. It reuses the existing SDK environment. `doctor` reports its
interpreter, dependency pins and manifest paths. For a custom plugin location,
set `OPENQUANTUM_SDK_ACCESS` to the absolute path of `sdk-access.mjs`.

For a separate simulator installation:

```sh
uv sync --frozen --no-editable --extra simulator
.venv/bin/vqa-joint run examples/three-spin-vqe.json --out runs/standalone --backend standalone
```

The simulator extra pins Qiskit, NumPy and SciPy; `uv.lock` records the resolved
dependency set. Without `uv`, create a Python 3.12 virtual environment and run
`python -m pip install .` or `python -m pip install '.[simulator]'` in it;
those commands use the declared direct pins rather than the full lockfile.
Choose a new output directory for each run.

## Use from Codex or Claude Code

### Install the Codex plugin

The GitHub marketplace bundles the Python wheel, a skill and the local
`vqa-compiler` MCP server. Install [uv](https://docs.astral.sh/uv/) first. For
the OpenQuantum backend, install OpenQuantum and prepare its
`hamiltonian-simulation` SDK.

```sh
codex plugin marketplace add 15735121580-jpg/wangximo-keyan
codex plugin add openquantum-vqa-harness@openquantum-research
```

Open **Plugins → OpenQuantum Research → OpenQuantum VQA Harness** in Codex
and start a fresh conversation. Invoke `$openquantum-vqa-harness` and ask it
to run the bundled three-spin VQE demo. The launcher uses the bundled wheel,
Python 3.12 and the existing OpenQuantum SDK. Its first start may download
Python and the pinned MCP dependency.

To add the same bundle to an existing local OpenQuantum plugin, point the
integration installer at the **source path** displayed by
`codex plugin list --marketplace personal --json`. Run these commands
from the `openquantum-vqa-harness` directory of your Git checkout:

```sh
python3 scripts/install_openquantum_plugin.py "/path/to/openquantum/plugin"
codex plugin add openquantum@personal
```

Use either plugin registration in Codex. Both expose the same harness
tools. To rebuild the wheel and bundled skill from this source, run
`python3 scripts/build_codex_plugin.py`.

### CLI and Claude Code registration

Install the package first, then register the stdio MCP server using that same
interpreter:

```sh
.venv/bin/python scripts/install.py --host codex
# Other choices: --host claude or --host both
```

The selected host CLI (`codex` or `claude`) must be on `PATH`. The installer
registers `vqa-joint-harness` and copies the `openquantum-vqa-harness` Skill to
`~/.agents/skills/` for Codex or `~/.claude/skills/` for Claude Code. It backs up
existing host configuration beside that configuration. Keep the checkout and
its virtual environment in place, then refresh the host's MCP connection.

The server provides:

| Tool | Purpose |
| --- | --- |
| `harness_status()` | Report the harness version and prepared OpenQuantum SDK. |
| `compile_vqa_job(job, output_directory, backend="openquantum")` | Compile and execute the supplied job; return results and artifact paths. |

MCP output directories must be absolute and new. `vqa-joint-mcp` runs the same
stdio server directly. Use the [agent prompt](docs/agent-prompt.md) with the
[example job](examples/three-spin-vqe.json); the installed Skill explains the
input and reporting contract. A host whose tool list has not refreshed can run
the CLI against the same input.

Codex or Claude Code supplies the model and its usual model authentication.
Local OpenQuantum SDK execution needs no separate model API key. Provider
credentials stay with the provider if remote execution is added later.
Host registration and an actual model-driven session are separate checks;
installing the Claude Code configuration alone does not establish a tested
Claude session.

## Read and reproduce a run

The run directory contains:

| Artifact | Contents |
| --- | --- |
| `input.json` | Complete task, device and sampling contract. |
| `compilation.json` | Selected rewrite, candidate costs, native programs and readout certificate. |
| `circuits.json` | Bound measurement settings, probabilities, fidelity and per-shot costs for both arms. |
| `host/terminal/*.qasm`, `joint/terminal/*.qasm` | Native terminal measurement circuits. |
| `strategy.md` | Executed strategy and selected terminal parameters. |
| `result.json` | Training records, fresh terminal answer, uncertainty, references, verification and full-job costs. |
| `runtime.json` | Selected backend and runtime metadata. |
| `replay.py` | Replay helper for an environment with the harness and simulator dependencies. |

Replay through the same backend with:

```sh
.venv/bin/vqa-joint run runs/demo/input.json --out runs/replay --backend openquantum
```

The seed, input hash, workflow hash and scientific package versions are
recorded. The exact full-space ground energy and ideal grid energies are
classical references; the reported VQE answer comes from sampled measurements.

## Scientific scope

Supported inputs use computational-basis product states, the implemented
Clifford/Pauli-rotation gate vocabulary, one real Pauli Hamiltonian, a connected
undirected device graph and at most eight device qubits. Pauli words and input
bits are **q0 first**; rotation angles are radians and physical times are
microseconds. See the [architecture and input contract](docs/architecture.md).

Certificates concern the declared ideal Pauli measurements on the reachable
input sector. They do not assert arbitrary-input unitary equivalence or
equivalent noisy channels. Search covers the implemented finite rewrite
catalogue and the supplied parameter grid. No hardware-wide global optimum is
claimed.

Execution is local simulation with supplied gate fidelities and independent
depolarization after native gates, including measurement-basis changes. Reset,
readout and memory are ideal in this model. Costs include all training and
terminal hardware shots, preparation, routing, basis changes, reset and readout;
offline compiler search is excluded from modeled execution time. A single
seeded comparison does not establish statistical superiority. Remote hardware
submission and active fault-tolerant encoding/decoding are outside this local
workflow.

MIT licensed; see [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
## Compiling a region supplied by an existing host

Use `vqa-joint compile-native job.json --out policy.json` to emit one policy for every binding of an existing host region. Agent clients can call `compile_native_host_policy` with the same JSON object. The host retains preparation, routing/transport, feedback, readout and stopping decisions unless the declared interface specifically allows changing one. Submit the emitted operations through the host's normal scheduler and carry any returned XOR through to its classical decoder.

Three interfaces are supported. The `settings` array contains `binding` (integer), `basis` (label) and `source` for every parameter/basis setting.

| Interface | Explicit host capabilities | Source |
|---|---|---|
| `cz-rotation-coherent` | Select one of `individual_u3` and `addressed_row_hrzhrx`, setting the other to false | Bound `num_qubits` and a native `gates` list of `gate`, `qubits` and `params`. The channel is compared modulo global phase. |
| `cz-rotation-readout` | `fresh_zero`, `terminal_z`, `retune_local_and_decode_xor` and `skip_whole_cz_run`; the first two must be true | Same native representation. The host must match disabled runs to complete independently switchable CZ batches and respect addressed groups. |
| `two-wire-rzz-relay` | `arbitrary_local_u`, `remote_cz` and `terminal_z`, all true | The existing front-end list of `[instruction, wires, angle_or_null]` over two wires with one RZZ region. Install the simulator extra for NumPy. |

The returned native words and source slots implement a deterministic rewrite. Full-job acceptance, native control feasibility, modeled process fidelity and elapsed time have distinct verification procedures. For an executor that returns shot blocks, `vqa_harness.bounded_selection.select_candidate` can skip the rest of a bounded-score training batch when its downstream decisions are already determined. This requires the score bounds, the host's tie rule and the ability to request another block after feedback.

### Guarding release against the original host

Run `vqa-joint guard-release packet.json --out decision.json` before outcome validation. The registered loss order sets the priority among eligible branches. The command preserves the original host whenever no certified feasible branch improves at least one protected loss while preserving the others. The JSON output includes the actual selected policy and the decision for each considered branch. A host with uncertain feasibility yields an unresolved decision. A proven infeasible host allows a separately labelled recovery.

The built-in model `native-phase-anf-v1` checks one ordered-register phase oracle with native X, Xdg, Z, CZ and CCZ instructions before routing. A packet contains exactly `model`, `losses` (a nonempty ordered list), `caps` (a resource-to-maximum object), `target` (wire_count and supports), `host` and `candidates`. Each policy contains wire_count and gates. Each scheduled gate has gate, qubits, position and layer, with positions and layers numbered from one. Recognized structural losses are gates, depth and active_wires. Missing bounds retain a feasible host. The verifier computes signed semantics, scheduling feasibility and costs from instructions before sampling.

For other models, call `vqa_harness.guarded_release.guarded_release(contract, host, candidates, verifier)` with a trusted verifier registered for that model. Its assess method certifies the hard constraints, and its upper_bounds method certifies each proposal-minus-original-host loss difference. Convert higher-is-better quantities such as fidelity to losses before registration. Deployment samples do not enter the release decision.

## Add the harness to a host program

An existing host can use the public Python interface in version 0.3.0. The client freezes its task, device, executable host with its controller, exposed stages and protected losses. It supplies the transformations for those stages and an independent verifier for the declared model.

```python
from vqa_harness import Branch, HostJob, ReleaseContract, compile_direct, compile_bounded

job = HostJob(task, device, original_program, exposed_roles, ReleaseContract(model, losses))
one = compile_direct(job, Branch("readout", ("measurement",)), adapter)
family = (Branch("readout", ("measurement",)), Branch("joint", ("routing", "measurement")))
searched = compile_bounded(job, family, budget=2, adapter=adapter, incumbent=one)
```

A2 keeps the delivered A1 executable unless a frozen candidate is certified nonworsening against both B and A1 on every registered loss, with a strict gain over A1. A named reward `J` is protected through the verified loss `-J`. Use `adapter.direct_branch(job)` to prescribe A1 when its receipt has not already been supplied. Original hardware acquisitions keep their contemporary policies. The source bundle includes the digest-pinned 0.2.0 wheel and `scripts/replay_v020.py` for archived comparisons.

For typed graphs, `direct_diagram` constructs one derivation by priority. `search_diagram` enumerates distinct complete diagrams under registered relation, depth and candidate limits. Both use the same verifier and return the selected policy with an audit. The optional Qiskit module `vqa_harness.readout_diagram` provides signed Pauli-effect transformations on bound circuits.

Run the two-round feedback client from an installed wheel with `python -m vqa_harness.examples.feedback`. It exercises all four entry points with a declared physical bit-flip model, then repeats them with a required chip-error bound unavailable to its verifier. [Host adapter interface and replay](docs/host-adapter.md)

For deployment on a device, use `qualify_saved`, `freeze_deployment` and `admit_deployment` with the physical contract and trusted evidence registry. The model compiler's selected candidate still needs independent physical acceptance. An unavailable physical answer bound withholds the candidate. Only a standby prequalified under the identical contract can be returned. [Deployment evidence and decision statuses](docs/host-adapter.md#qualifying-a-frozen-policy-for-physical-deployment)
