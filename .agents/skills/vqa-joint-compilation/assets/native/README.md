# VQA Joint 0.1.0a5

VQA Joint accepts measured, numerically bound circuit packets and produces native schedules, a reusable host strategy and a per-shot decoder. Application stages share one action across the supplied angle bindings. The default objective minimizes total native window time for the packet's declared serial executions, with gate count and peak data atoms as tie breakers.

## Install

Use Python 3.10+ for the CLI. From the delivery directory, run the following in a fresh virtual environment.

```sh
python -m pip install ./vqa_joint-0.1.0a5-py3-none-any.whl
vqa-joint --help
```

Alternatively unzip the source archive and run `python -m pip install .` in its project directory. The distribution requires NumPy. Each upstream compiler runs in its own interpreter. The example below prepares both hosts with Python 3.12 using their verified revisions.

```sh
git clone https://github.com/BAQIS-Quantum/neutral-atom-compilation.git zap-source
git -C zap-source checkout ba75e945da6c14ae41ee0495dbc8e26fa4e11f8e
python3.12 -m venv zap-env
zap-env/bin/python -m pip install -r zap-source/requirements.txt 'qiskit==2.6.2'
export JOINT_ZAP_SOURCE="$PWD/zap-source" JOINT_ZAP_PYTHON="$PWD/zap-env/bin/python"

git clone https://github.com/UCLA-VAST/ZAC.git zac-source
git -C zac-source checkout e5083362f99e6915f20c2bd0eaa88b6d1cdcecac
python3.12 -m venv zac-env
zac-env/bin/python -m pip install -r zac-source/requirements.txt
export JOINT_ZAC_SOURCE="$PWD/zac-source" JOINT_ZAC_PYTHON="$PWD/zac-env/bin/python"
```

## Compile, sample, decode and reuse

The example prepares a state on logical wire 2 and entangles it with wire 0. Wire 1 remains zero. For the one-site implementation, measured native bit 1 decodes to logical string 101. Each output directory must be new.

```sh
vqa-joint compile example.json --host zap --out zap-run
vqa-joint compile example.json --host zac --out zac-run --goal peak_data_atoms,duration_us
bit=$("$JOINT_ZAP_PYTHON" sample_native.py zap-run/result.json --stage readout --binding theta0)
vqa-joint decode zap-run/result.json --stage readout --binding theta0 --native "$bit"
vqa-joint replay example.json --strategy zap-run/strategy.json --out zap-rebound
```

The helper draws one seeded ideal statevector sample from the emitted angle-bound native gates. A device driver can supply its measured little-endian integer directly to `decode`. For randomized readout kernels, the CLI draws the required independent bits and records them. Use `--random-bits` to reproduce a specific draw. Replaying accepts new numerical angles when the task contract, stage shapes and emitted actions satisfy the stored guards. A refused binding produces a nonzero status.

## Read an optional FluxQ intent

The companion `fluxq_packet.py` runs under the external FluxQ 0.2.0 interpreter. It converts a fully bound QSpec circuit and its canonical readout to one CLI packet. The upstream four-wire ring example exercises the conversion. Run these commands from the delivery or extracted source directory.

```sh
git clone https://github.com/xi-zhao/FluxQ.git fluxq-source
git -C fluxq-source checkout e040367d0a8b06fdd44d161369434c220286d48f
python3.11 -m venv fluxq-env
fluxq-env/bin/python -m pip install './fluxq-source[qiskit]'
fluxq-env/bin/python ./fluxq_packet.py fluxq-source/examples/intent-qaoa-maxcut.md packet.json
vqa-joint compile packet.json --host zap --out fluxq-zap
```

For another compatible source intent, pass its path in place of the upstream example. The bridge requires a new output file and checks width, depth, gate semantics, terminal readout and finite parameters. Inputs with explicit hardware connectivity require a matching target adapter. OpenQuantum and QwenQPU can supply additional independent checks through their respective runtimes.

## Input and search

Use the schema in `example.json`. Each stage and binding has an ID. A binding begins in the all-zero state on 1–8 wires and carries a positive number of shots. Each operation specifies its gate name, distinct in-range operands and angle in radians, with `null` for fixed gates. Supported gates are H, X, Y, Z, S, Sdg, T, Tdg, RX, RY, RZ, CX, CZ and RZZ, all written in lower case. End with one indexed Z measurement per wire in increasing order. To express uniform X measurement, apply H to each wire first.

The joint pass considers the original circuit, equivalent parity decompositions, and structurally recognized measured alternatives. These include global-flip-invariant plus-prepared RX/RZZ circuits and a single active input followed by a CNOT network with full Z or uniform X readout. Arbitrary input wire and CNOT-derived readout masks are supported. Pass `--wire-relabel` to include the declared wire permutations. `--original-only` disables rewrites. The default uses one upstream profile. Select multiple profiles with repeated `--profile`, or search the six ZAP or four ZAC profiles using `--all-profiles`.

The cost order is comma-separated and lexicographic. Candidates minimize any distinct subset of `duration_us`, `gate_count`, `peak_data_atoms`, `gate_dependency_depth` and `gate_loss_proxy_sum`. Sum duration, gate count and gate dependency depth over declared shots. Take the maximum data atom count over serial windows. For a window with gate counts n1 and n2, the gate proxy adds shots × (1 − 0.9999^n1 × 0.999^n2). The two constants are fixed single- and two-qubit average-fidelity assumptions. This independent-gate resource proxy has no calibrated interpretation as process fidelity. `--budget` caps the number of evaluated strategy combinations after native lowering. The result reports explored and eligible counts. An exhaustive result applies to the supplied candidate envelope.

## Outputs and evidence

`strategy.json` stores profile, rewrite family, implementation, wire order, shape guards and evaluation scope. `result.json` links each selected binding to its QASM, angle-bearing gate sequence, upstream placement/movement trace and metrics. Native metrics include the ideal operator discrepancy and adapter geometry checks. A run with zero eligible actions exits with status 2, records rejections in `result.json` and emits no strategy. Existing directories are refused. Read the manifest's status before consuming schedules.

The CLI evaluates serial unitary-window resource subtotals for the submitted shot counts. It accepts bindings from an external variational controller. The caller runs the optimizer and accounts for classical feedback, external preparation/reset, terminal readout, QEC and physical noise. Actual whole-job time, process fidelity and decoded logical failure remain unknown. The pinned ZAC one-site specialization calls its native router for an empty entangling layer. ZAC checks timing dependencies and upstream scheduling/placement. ZAP also checks discrete AOD and occupancy constraints. These are software checks on specific host models.

## Verify the installation

From the extracted source directory, use the CLI environment's Python to run `python check_install.py --output installation-check.json`. The test compiles the sparse-mask example on both actual host programs, decodes one ideal native shot, replays a changed angle and checks refusal paths, including IDs with path separators. Configure both external hosts as above.

## Two-word affine instruments and fixed-weight programs

For a variational circuit with a single active input, an X/CNOT encoder and terminal local X, Y or Z bases, the CLI can retain one native quantum wire and reconstruct every requested logical bit string with a conditional affine binary kernel. Set the feedback packet's `circuit_process_port` to `affine_two_word` to audit the coherent isometry with an open input. Fresh product spectators and terminal classical output define this interface. For X or Y readout, write H or Sdg then H ahead of the indexed terminal measurement. Downstream coherent operations on the materialized full register require another interface. The supplied H₂ example is one fixed-angle XXYY readout stage with a sixteen-shot parity criterion.

```sh
vqa-joint compile affine_example.json --host zap --goal peak_data_atoms,gate_count,max_circuit_process_infidelity --out affine-run
vqa-joint replay affine_example.json --strategy affine-run/strategy.json --out affine-rebound
vqa-joint sample-job affine-rebound/result.json --seed 0
```

The SDK class `vqa_joint.sector_qaoa.fixed_weight.FixedWeightPort` compiles fixed-cardinality diagonal-cost/XY-exchange programs. Declare the seed mask, preparation exchanges, diagonal energies and ordered mixer layers. Its compact register has ceil(log2(binomial(n,k))) quantum wires, with padded measurement labels reported as unavailable. The default dense-audit budgets admit source dimension up to 256 and fixed-weight dimension up to 64. Request the optional Qiskit synthesis dependency with `python -m pip install '.[synthesis]'` from the extracted source tree. Then run `python fixed_weight_example.py` to synthesize the source, the full-width optimized realization and the rank-encoded candidate and verify their phase-aligned coherent maps. The example is a three-mode one-particle problem. Fixed-weight compilation uses the Python SDK entry point. Its returned QASM can be supplied to an appropriate host interface. Matrix-level verification alone determines neither physical speed nor optimal native placement.

The wheel includes both new examples. To copy them into the working directory after a wheel-only installation, run the following with the CLI virtual environment.

```sh
python -c 'from pathlib import Path; from importlib.resources import files; base=files("vqa_joint")/"examples"; [(Path(name).write_bytes((base/name).read_bytes())) for name in ("affine_example.json", "fixed_weight_example.py")]'
```

## Rebuild the wheel

From the extracted source directory, run the following. SOURCE_DATE_EPOCH fixes the archive timestamp. The build configuration pins the build backend.

```sh
SOURCE_DATE_EPOCH=1704067200 python -m pip wheel --no-deps . --wheel-dir dist
```

## Measurement-conditioned jobs

The feedback_example.json packet implements a complete triangle MaxCut QAOA controller. It evaluates two batches of 32 measurements and then returns the best observed partition from 64 fresh shots. The second probe and final parameter both depend on earlier measurements. Integer return rank encodes the cut in its quotient by eight and the partition as seven minus the remainder. Run the same file with and without rewriting on either ZAP or ZAC.

```sh
vqa-joint compile feedback_example.json --host zap --out original --original-only
vqa-joint compile feedback_example.json --host zap --out joint
vqa-joint sample-job joint/result.json --seed 42
vqa-joint replay feedback_example.json --strategy joint/strategy.json --out rebound
```

The evaluator propagates the finite distribution over live classical registers. The gate model uses independent local depolarization calibrated to average gate fidelities 0.9999 and 0.999. Runtime callbacks can supply physical outcomes through FeedbackProgram.run. Whole-job model costs include only the specified mechanisms. Duration is the sum of native unitary windows weighted by actual branch probabilities. A complete serialized timing scenario additionally requires six explicitly supplied clock terms in feedback_job.CLOCK_TERMS. Modeled peak data atoms exclude auxiliary device inventory. Changing the feedback graph, shot budgets, noise contract or success threshold requires fresh policy extraction.

The optional developer adapter vqa_joint.new_host.powermove.power_host uses the original PowerMove repository at commit abb420504b65276653220e385d31b75056f65912. Set JOINT_POWERMOVE_SOURCE to this checkout and use a Python environment with compatible Qiskit, networkx and matplotlib. PowerMove reports transport_us separately from total execution time.

## Host responsibilities and paired comparison

`vqa_joint.host_scope.build_scope` declares the stages covered by a host and supplies conventional protocols for the other applicable stages. Passing a concrete protocol ID records its version. The register covers the ten digital research hosts and the Pulser analog host. Caliper is an optional factory component. A declared protocol still needs appropriate executable bindings and evidence for a particular device. Static packets cover their declared calls. Finite feedback packets also execute the classical decision graph and evaluate final delivery.

For packets compiled using this version, compare the two selected arms directly.

```sh
vqa-joint compare original/result.json joint/result.json
```

The command checks the complete input packet, host, profile envelope, noise conditions, stage responsibilities and order of objectives. It returns both measured model scores and the changes in selected actions. The enumerated strategy coverage refers to the supplied candidates and evaluation model. The developer API also permits explicit cross-stage rewrite relations and stage-level source records.

## Verified process ports and whole-job selection

The process_example.json packet specifies a common two-dimensional coherent port for the symmetric triangle, a circuit fidelity floor, a complete-answer success floor and an explicit serial timing scenario. Compile it with the ordered goal modeled_job_time_us,gate_count,peak_data_atoms,max_circuit_process_infidelity. The compiler evaluates the original reference and filters candidate policies for non-regression on every available modeled endpoint before ranking them. Whole-job execution on physical hardware still needs a calibrated timing and noise model. The energy or cut score is defined by the caller's task contract.

For an all-plus circuit with Rx, Rz, Rzz and CNOT, the binary_parity_first port derives a common Hilbert space from the logical phase parities. Candidates outside the required coherent representation cannot pass its Choi check. The symmetric three-vertex graph additionally admits two permutation/complement orbits and a uniformly randomized readout. The executable kernel uses unbiased rejection sampling for the orbit with six labels and charges its corresponding average fair-bit service. Lowering and search can feed existing hosts without changing their native implementation. The CLI is backed by ZAP and ZAC; research adapters and paper-protocol reproductions have individual runners.

The protected outcome fields for analog pulse workflows include native pulse count and depth. Replaying a rewritten policy rebinds both the selected and original action at the new parameters, evaluates the task threshold and rejects regressions in the matched available model costs. For legacy strategies without a saved original action, replay searches the same original host profile envelope (up to 256 combinations) and records the selected reference. An unchanged strategy is verified against its job thresholds. These tests concern the declared model endpoints.

The SDK function `vqa_joint.terminal_controller.best_of_shots` submits a finite terminal measurement in blocks. Its caller supplies the outcome ranking and an independently justified certificate that the incumbent is the highest-ranked feasible outcome, including the tie rule. The controller returns the same best answer as a full fixed batch on a shared shot stream and can skip all later blocks once certified. An output contract requiring a full sample histogram must use its entire batch. The source project includes release tests runnable after installation with `python -m unittest discover -s tests -v`.

## Closed-loop encoded ZAC VQE

```sh
vqa-joint compile-encoded encoded_example.json --out encoded-run
vqa-joint replay-encoded encoded-run/result.json --seed 42
```

The wheel carries the archived native gates, ZAC placement/motion traces, QASM, gate/depth/timing metrics and the published device specification for both preparations. With NumPy installed the commands run offline. The example is also distributed as `vqa_joint/examples/encoded_example.json`. The stock ZAC hardware specification drives the recorded geometry/timing. For physical gate errors the test uses one-/two-qubit average fidelities of 0.9999/0.999, reset/readout flips of 0.001, and ideal motion and memory. Additional fresh-reset, parallel-image, decoding and feedback costs are fixed serial scenario inputs.

Set `host` to `ZAC`, `task` to `steane_grid_vqe` and `train_shots_per_basis` to 16 or 64. The task minimizes −X − 0.75 Z on the three angles 0 and ±π/2. Both bases are measured, a conditional training comparison can open a third evaluation, and a separately sampled energy determines full-job success. The compiler enumerates 64 assignment maps, each choosing seven-atom destructive heralding or fourteen-atom verified preparation at each of six parameter/basis calls. Constraints include `min_application_success`, `min_preparation_fidelity`, `min_answer_distribution_fidelity`, `min_native_gate_process_fidelity`, `max_peak_atoms`, `max_expected_gates`, `max_expected_depth`, `max_expected_time_ms`, `max_decoded_flip_fraction`, `max_coupled_job_flip_probability` and `require_verified_preparation`. Reviewed numerical floors and ceilings identify admissible policies before ranking by complete expected time. Time, gates and depth include retries.

The emitted `result.json` identifies the individual evidence fields, all 64 assignments, their known violations and any unknown constraint, and the selected model policy when one is certified. `optimal_in_catalogue` establishes fastest among the finite set under the given model. `best_certified` indicates a certified winner while a faster candidate has unresolved evidence. `infeasible_in_catalogue` and `uncertified` exit with code 2 after saving the reasons. For a selected policy, six original ZAC schedules with their metrics, geometry traces and QASM can be found via `native-files.json`. The geometric description and its upstream BSD licence are supplied with those traces. Replay rechecks the archived policy and simulates the native Clifford gates, retries, classical decisions and delivered energy from a fresh seed.

The state-fidelity floor protects fresh logical-state preparation at the pre-readout port. `min_native_gate_process_fidelity` concerns the native unitary window on its own physical inputs before reset or readout. A requested common arbitrary-input logical circuit process floor (`min_circuit_process_fidelity`) or repeated nondestructive round-error limit (`max_round_logical_error`) requires additional calibration/evidence and returns an unknown constraint. The returned decoded error refers to accepted logical signs.
