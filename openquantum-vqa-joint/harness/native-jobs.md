# Native compilation in the harness

With the Skill's native dependencies installed, run `python harness/run.py job.json --structured --out run` from the directory containing `harness/`. Each output directory must be new. A `mode` of `native` calls the bundled VQA Joint compiler. Supply the complete circuit packet as a JSON object in `packet`.

- `host: "zap"` and `host: "zac"` use the configured upstream native compilers and produce angle-bound schedules. Install the pinned third-party source and configure its interpreter as described in the [native compiler instructions](../../.agents/skills/vqa-joint-compilation/assets/native/README.md). Optional fields are `goals` (ordered list), `profiles` (nonempty list for the chosen host), positive integer `budget`, and boolean `wire_relabel` and `original_only`. The full native feedback packet, including its clock and noise contract, is accepted. A `replay_packet` rebinds the strategy to another compatible packet. A nonnegative `seed` additionally samples the original compiled feedback job.
- `host: "zac-encoded"` runs the offline Steane VQE catalogue. A `seed` replays the selected policy locally. This catalogue uses its own encoded-task contract.

Example preparation from the repository root uses the distributed full input rather than a path supplied to the model:

```sh
python - <<'PYTHON'
from pathlib import Path
import json
src = Path('.agents/skills/vqa-joint-compilation/assets/native')
packet = json.loads((src / 'encoded_example.json').read_text())
Path('job.json').write_text(json.dumps({'mode':'native', 'host':'zac-encoded',
                                       'packet':packet, 'seed':27}))
PYTHON
python openquantum-vqa-joint/harness/run.py job.json --structured --out encoded-run
```

For an unencoded ZAP or ZAC run, set the corresponding host, use the bundled `example.json` as the packet, and omit `seed`. The output manifest links to the native report and to any requested replay. Inspect the scope, objective and candidate counts in that report. A nonzero exit status identifies rejected inputs, an infeasible job or a failed native operation.
