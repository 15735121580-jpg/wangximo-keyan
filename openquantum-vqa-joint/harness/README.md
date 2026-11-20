# Model-to-compiler harness

This harness adapts a Codex model to the joint compiler through a fixed structured contract. The model transcribes supplied values. The Python compiler checks the physical model's declared resource and quality limits, solves the finite problem, verifies the optimum and replays a seeded two-round feedback VQE job.

The adapter was exercised with GPT-6.1 Sol in a new ephemeral Codex session with memories disabled. The extracted quantitative contract and compiler receipt matched an independent calculation. The default entry point handles the two-round Rb–Cs input. Other structured backend tasks can use the bundled Skill's general finite and interval solver APIs.

From the repository root, prepare the environment and run the included synthetic request in a new output directory.

```sh
python -m venv .venv
.venv/bin/python -m pip install -r .agents/skills/vqa-joint-compilation/assets/certified/requirements.txt
.venv/bin/python -B openquantum-vqa-joint/harness/run.py \
  openquantum-vqa-joint/harness/validation/intent.txt --out harness-run
```

The executable needs Codex CLI with a configured provider and model that supports structured output. `--model` changes the parser model. For a local source-tree package, use `--skill` to identify a separate Skill checkout if needed.

Inspect `contract.json`, `result.json`, and `provenance.json` for the interpreted input, exact-solver receipt and run provenance. The model can return a missing-input response. Verify extracted numbers and hardware assumptions against their sources before interpreting the result. The model interface does not change the pretrained weights. The released validation is an offline synthetic architecture comparison.
