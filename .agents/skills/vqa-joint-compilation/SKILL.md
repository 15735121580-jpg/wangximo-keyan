---
name: vqa-joint-compilation
description: 为变分量子作业比较完整编译策略，支持 Rb–Cs 双组分原子阵列的用户成本合同、有限或区间模型求解、ZAC 编码策略回放和 ZAP/ZAC 编译。用于给定任务、架构候选及约束后选择策略并本地小规模验证。
---

# Variational quantum joint compilation

Match every candidate to the same task and report which program, hardware model, budgets and execution window each comparison uses. Read [the input contract](references/execution.md) for the selected mode and run the corresponding bundled entry point with the host's local code tool.

- For an Rb–Cs array and the two-round feedback VQE task, translate the supplied calibration or simulator outputs into a complete branch-time packet and call `scripts/compare_two_species.py`. Use `assets/two-species-vqe.json` for an explicitly synthetic offline demonstration. Treat geometry and native instruction legality as inputs from the proposed backend, with their evidence status attached.
- For a new feedback job, use the SDK in `assets/certified` with a typed action graph, observable outcome labels, complete branch costs and resource constraints. Choose the exact or interval solver as specified in the reference.
- For the archived encoded ZAC task, use `compile-encoded` and `replay-encoded` from the bundled `vqa-joint` source. For new angle-bound circuit packets, use the ZAP or ZAC compile/replay interface with its configured native compiler. These backends have separate hardware contracts from the Rb–Cs model.

Run the calculation and return the actual receipt path when local code execution is available. Describe the winner, tie or failed bounds and report the metric units. A certificate is relative to the supplied catalogue, available physical schedule and declared model. Keep the squared fidelity of a classical readout record, native unitary process fidelity and measured logical error separate. Ask for a missing experimental input when it affects the selection, or use clearly labelled fixture values if the user requests an example.

Use only OpenQuantum tools that the current host has actually enabled, for independent simulation or physical validation when requested. This Skill's bundled scripts run locally through the host's code tool and leave a JSON result. Its installation and upstream contribution paths are in [integration](references/integration.md).
