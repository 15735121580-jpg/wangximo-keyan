# Model-to-compiler harness

This harness adapts a Codex model to the joint compiler through a fixed structured contract. The model transcribes supplied values. The Python compiler checks the physical model's declared resource and quality limits, solves the finite problem, verifies the optimum and replays a seeded two-round feedback VQE job.

The adapter was exercised with GPT-6.1 Sol in a new ephemeral Codex session with memories disabled. The extracted quantitative contract and compiler receipt matched an independent calculation. The default entry point handles the two-round Rb–Cs input. For other backends, structured JSON dispatches typed feedback diagrams and native requests to the packaged compiler.

From the repository root, prepare the environment and run the included synthetic request in a new output directory.

```sh
python -m venv .venv
.venv/bin/python -m pip install -r .agents/skills/vqa-joint-compilation/assets/certified/requirements.txt
.venv/bin/python -B openquantum-vqa-joint/harness/run.py \
  openquantum-vqa-joint/harness/validation/intent.txt --out harness-run
```

The executable needs Codex CLI with a configured provider and model that supports structured output. `--model` changes the parser model. For a local source-tree package, use `--skill` to identify a separate Skill checkout if needed.

Inspect `contract.json`, `result.json`, and `provenance.json` for the interpreted input, exact-solver receipt and run provenance. The model can return a missing-input response. Verify extracted numbers and hardware assumptions against their sources before interpreting the result. The model interface does not change the pretrained weights. The released validation is an offline synthetic architecture comparison.

## Structured compiler jobs

`--structured` accepts JSON directly and uses the same packaged compiler sources as the Skill. A provider login is needed only for the natural-language entry point above. Three modes are available.

- `finite` selects a complete policy and optimizes its bounded real controls using the exact solver. It rechecks both feasibility of the selected control and the absence of a strictly shorter feasible policy in the declared catalogue.
- `interval` passes certified rational enclosures to the interval solver, rechecks the final covering partition and feasible witness, and reports lower/upper bounds and the achieved gap.
- `native` invokes the packaged native compiler on a structured host request. See [native requests and execution](native-jobs.md).

From the repository root run the bundled finite feedback example:

```sh
.venv/bin/python -B openquantum-vqa-joint/harness/run.py --structured \
  openquantum-vqa-joint/harness/validation/typed-job.json --out exact-run
.venv/bin/python -B openquantum-vqa-joint/harness/test_typed_jobs.py -v
```

A finite or interval job specifies the task, input and output port types, common time unit and a list of policies. Each policy declares bounded controls, a root and a directed acyclic graph. Nodes have input/output port types, a duration and a map from observable record labels to `[probability, successor_id]`. A terminal has an empty map. The harness checks ports, reachability, acyclicity and exact normalization, then forms the complete expected time by backward composition over every branch. Durations must be nonnegative at a feasible witness. The user supplies the task semantics, hardware evidence and all additional whole-job constraints as polynomial residuals that must be at most zero. A shared port name checks type compatibility; the caller establishes that policies implement the same task. For a fixed node, declared branch probabilities are constants independent of the controls. The underlying Python SDK also supports richer user-supplied formulas and enclosure functions.

Use integers or quoted rational or decimal constants. Write `["var","x"]` for a control and `[op,left,right]` for polynomial operations `+`, `-`, `*`. Control bounds are inclusive. To run the example in interval mode, set `"mode": "interval"`, add `"tolerance": "1/32"` in the stated time unit and `"max_rounds": 8`. The two candidate policies have minimum times 10 and 7 µs. Control `x = 1` realizes the latter, including the 1/4-probability slow branch. Their numbers describe a synthetic complete-job timing contract.

The fresh output directory contains the original `job.json` and a `result.json` for well-formed solver runs. Optimal time and control values in finite mode use Z3 exact number syntax. Interval receipts include rational bounds and the complete cover used for verification. A well-formed infeasible catalogue or an unresolved interval gap exits with status 2 and its receipt. Invalid port graphs and probability kernels exit with status 2 before certification. A zero exit status denotes a verified exact or tolerance-qualified solution for the supplied catalogue. The natural-language parser currently translates the two-round Rb–Cs description; users supply general typed and native jobs directly as JSON.
