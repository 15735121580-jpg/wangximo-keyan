# Running a comparison

Use Python 3.10 or newer from an environment with the required packages. The commands below assume the working directory is the skill folder or that `skill_root` points to it. In the command examples, set `run_dir` to an existing writable directory for the new results. Separate use of a virtual environment is optional when the required dependencies are already available.

## User-defined Rb–Cs transfer table

Set `data_species` and `service_species` to different elements from Rb and Cs. State which values came from simulation, calibration, or a synthetic fixture. Convert times to microseconds. Include the setup and readout costs through the first observable outcome in `prefix_us`. Include all remaining physical and online controller time in each observable branch of `suffix_us`. Record both species' peak inventories including service atoms, the maximum total branch depth, and an independent symmetric readout-flip rate for the candidate. A pre-measurement choice of architecture stays fixed on both of its branches. The packet accepts a complete set of declared candidates. Their geometry, collision constraints, measurement and reset scope, qubit to atom mapping, compatible native gates, noise assumptions, and applicable QEC actions must be established separately.

The bundled task samples parameter A first. It measures logical Z and then evaluates B after visible +1 or A after visible −1. It reports the parameter with lowest observed mean energy. Its ideal positive Z probabilities at A and B are 3/4 and 1/4. The supplied scalar is the independent symmetric flip probability per recorded readout. The comparison tests the probability of at least one model readout flip and the squared classical fidelity of the two-readout records. Gate errors, loss, leakage, transport and correlated flips require additional terms or a separate backend. The summed per-species peaks are a conservative bound if the individual peaks occur at different moments.

Read `assets/two-species-vqe.json` as the complete editable packet. Use integers or decimal/fraction strings for exact scalar fields. Every candidate must represent the same two-round semantics. A device with a different measurement model or controller should use the SDK below. A missing calibrated logical process fidelity cannot be substituted by record fidelity.

```sh
python -m pip install -r "$skill_root/assets/certified/requirements.txt"
python -B "$skill_root/scripts/compare_two_species.py" \
  "$skill_root/assets/two-species-vqe.json" --seed 42 --out "$run_dir/two-species.json"
```

This fixture selects `feedback-aware` at 83/10 µs compared with 49/5 µs for `short-prefix`. The finite catalogue is checked for feasibility and for any strictly shorter feasible candidate. The optional seed requests one modeled record and executes the actual classical decision rule. The result stores all candidate metrics, bound checks, tied minimizers and replay. An infeasible catalogue writes its receipt and returns exit status 2. A malformed input also returns status 2 with an error message. Existing result files are refused.

## Certified feedback and control SDK

Add the bundled `assets/certified` directory to the module search path of a workspace script. For a new task, use `policy_ir.py` to generate legal typed policies and lower each policy to an `optimal_compiler.Diagram` with its complete Boolean feasibility formula, real time objective and program witness. The exact solver checks a finite catalogue whose constraints and costs are semialgebraic over the reals, using Z3. Re-run `verify` on the declared catalogue and the result. For a domain with sound convergent interval extensions, use `interval_compiler.solve` and `interval_compiler.verify`. The report must state the domain, bound gap, and any unattained or unresolved status. `feedback_vqe.py`, `full_qaoa.py` and `smooth_vqe.py` provide replayable examples. Inferring a device-wide optimum requires an independently complete catalogue, defensible control bounds and certified device feasibility.

The bundled sources are the maintained reference solver and tests. From an editable copy of `assets/certified`, run `python -B -m unittest discover -v` with the supplied Z3 version. For other models, preserve any distinct loss, retry, fidelity, and online timing definitions from the input. Consult the user's original hardware evidence before extending the catalogue.

## Native compilation and archived encoded replay

Install the packaged source into an available Python environment. The installation may need access to Python package dependencies. Once available, the archived example and replay run offline with no credentials or upstream ZAC executable.

```sh
python -m pip install "$skill_root/assets/native"
vqa-joint compile-encoded "$skill_root/assets/native/encoded_example.json" --out "$run_dir/encoded"
vqa-joint replay-encoded "$run_dir/encoded/result.json" --seed 42 > "$run_dir/encoded-replay.json"
```

The solver checks 64 assignments of seven-atom destructive heralding or fourteen-atom verified preparation to six ZAC parameter/basis calls under an explicit circuit-noise model and serial clock. The replay advances the encoded circuit, retries, decision rule and fresh answer readout. Preserve the `status` fields and whether a requested constraint has evidence, especially for an arbitrary-input channel or repeated nondestructive syndrome error. Read `assets/native/README.md` for artifact schemas and for the pinned ZAP/ZAC installation steps for new angle-bound circuits. Native scheduling and sampling scopes depend on the backend. A different neutral-atom species contract requires the corresponding native adapter and evidence.
