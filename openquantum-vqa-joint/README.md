# Variational quantum joint compilation for OpenQuantum

This project-local Skill compares complete feedback policies under the supplied hardware and task contract. It includes exact and interval model-based synthesis, ZAP and ZAC native adapters, a ZAC encoded catalogue and an offline Rb–Cs feedback VQE example.

With OpenQuantum installed in Codex, open the repository in a fresh conversation and invoke `$vqa-joint-compilation`. The complete Skill is in [`.agents/skills/vqa-joint-compilation`](../.agents/skills/vqa-joint-compilation/SKILL.md). See [execution instructions](../.agents/skills/vqa-joint-compilation/references/execution.md) for setup and [integration instructions](../.agents/skills/vqa-joint-compilation/references/integration.md) for OpenQuantum distribution.

## Replay the Rb–Cs validation

From the repository root, use Python 3.10 or later. In a virtual environment install the pinned solver and run the user-defined contract in microseconds.

```sh
python -m venv .venv
.venv/bin/python -m pip install -r .agents/skills/vqa-joint-compilation/assets/certified/requirements.txt
.venv/bin/python -B .agents/skills/vqa-joint-compilation/scripts/compare_two_species.py \
  openquantum-vqa-joint/validation/contract.json --seed 27 --out fresh-result.json
```

The validated synthetic catalogue has two feasible architectures. The selected branch-local policy takes 8.3 µs in expectation, compared with 9.6 µs, with 8 versus 10 peak atoms and maximum depth 12 versus 13. Seed 27 yields readouts (−1, +1), uses parameter A twice and selects A with mean energy 0. `validation/result.json` and `validation/verification.json` record the local result and independent check. The optimality claim applies to the supplied synthetic catalogue.

To independently verify the bundled receipt, run the following in a new temporary directory after copying or linking the repository's `.agents` folder there.

```sh
cp /path/to/repository/openquantum-vqa-joint/validation/{contract.json,result.json,verify_result.py} .
/path/to/repository/.venv/bin/python -B verify_result.py
```

The original Skill and compiler code are distributed under the MIT License. The bundled ZAC architecture retains its [upstream licence](../.agents/skills/vqa-joint-compilation/assets/native/src/vqa_joint/packaging/ZAC-LICENSE.txt).
