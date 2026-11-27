---
name: openquantum-vqa-harness
description: Given a variational circuit, output observables and hardware interface, use OpenQuantum to compile, verify and execute a complete local VQA job, returning optimized native programs, an execution strategy and the answer.
---

# OpenQuantum VQA joint harness

For certified joint compilation on a registered physical adapter, call
`compile_joint_job(task, platform, tolerance)` or `vqa-joint compile-joint`.
Use the platform's objective unit and an exact absolute tolerance. Return the
programme in `policy` for `optimal` or `epsilon_optimal`, with scope and bounds.
An `open` result gives a diagnostic incumbent and interval; its delivery field is
empty. For a finite-grid VQA calculation with a sampled answer, call
`harness_status` then `compile_vqa_job` from the plugin's `vqa-compiler` server
(or the legacy `vqa-joint-harness` server). Read the bundled
`assets/three-spin-vqe.json` for the input schema and reproducible example. In a
source checkout it is also available at `examples/three-spin-vqe.json`.
Construct the input from the user's task and device.
Keep supplied circuits, coefficients, bit order, units, budgets and fidelities.
Ask about missing physical data when the user requests a calibrated device.
For an explicitly requested demonstration, label the supplied device as a model.

Pass an absolute, new output directory and `backend="openquantum"`. The harness
discovers the pinned Qiskit SDK through OpenQuantum's `sdk-access.mjs`. The host
model provides reasoning and MCP calls. Its existing Codex or Claude login
supplies model access. The local OpenQuantum SDK requires no model API key.
Remote provider authentication, if needed in future work, stays with its provider.

The workflow retains the variational ansatz and parameter grid, discovers
backward Pauli and affine-sector rewrites from their typed diagram, selects a
native implementation and independently checks ideal output expectations. It
executes initialization, gate decomposition, mapping/routing, scheduling,
Pauli-basis measurement, finite-shot training, fresh terminal readout and
classical energy reconstruction. Encoding or decoding requires its own declared
interface. The example specifies physical qubits with no active encoding.

Read the actual `result.json`, `compilation.json`, `strategy.md`, native circuit
files and equivalence residuals. Report the selected pass, complete-job costs,
sampled answer and its uncertainty, reference value and applicability. Compare
ideal semantics separately from the declared noisy execution. The finite rewrite
catalogue determines the search coverage and certificates. Retain ties and
regressions that the measured answer shows.

If the host has not refreshed its MCP tool list, use the installed `vqa-joint`
CLI with the same input and OpenQuantum backend, and state that execution path.
Save job inputs and commands. Do not request a token in chat or read a credential
store. Return concrete file links and a readable full strategy in the conversation.
