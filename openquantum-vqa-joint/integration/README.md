# Build the joint compiler into the OpenQuantum Codex plugin

The integration patch targets [OpenQuantum commit 41e12b8](https://github.com/xi-zhao/OpenQuantum/tree/41e12b899429d575ae102ccb9b86ef56d03b76c9). It registers one L0 capability and includes the version-controlled compiler files alongside its generated Codex Skill. The model executes scientific code using the host’s local code tool. A fresh checkout keeps the generated plugin separate from installed releases.

From the root of this joint compiler release, with Git, Node 24+, npm and Python 3.10+ available:

```sh
release_dir="$PWD"
git clone https://github.com/xi-zhao/OpenQuantum.git OpenQuantum-joint-build
cd OpenQuantum-joint-build
git checkout --detach 41e12b899429d575ae102ccb9b86ef56d03b76c9
git apply --check "$release_dir/openquantum-vqa-joint/integration/openquantum.patch"
git apply "$release_dir/openquantum-vqa-joint/integration/openquantum.patch"
cp -R "$release_dir/.agents/skills/vqa-joint-compilation" .agents/skills/
git add .agents/skills/vqa-joint-compilation .agents/capability-packages.yml scripts/build-chatgpt.mjs
npm ci
npm run capability:conformance
node scripts/build-codex-plugin.mjs
```

The release is in `dist/codex-plugin/`. From that same checkout, exercise the packaged Skill directly after extracting the zip in a writable temporary directory:

```sh
replay_dir=$(mktemp -d)
unzip -q dist/codex-plugin/openquantum-*-codex-plugin.zip -d "$replay_dir"
projected="$replay_dir/openquantum/skills/vqa-joint-compilation"
python3 -m venv "$replay_dir/venv"
"$replay_dir/venv/bin/python" -m pip install -r "$projected/assets/certified/requirements.txt"
"$replay_dir/venv/bin/python" -B "$projected/scripts/compare_two_species.py" \
  "$release_dir/openquantum-vqa-joint/validation/contract.json" --seed 27 --out "$replay_dir/result.json"
```

The [integration validation](validation.json) records the 199-package upstream conformance audit, the 191-Skill release build, the real stdio discovery response and the extracted-archive replay. All 134 bundled resource files checked against their source. The test checkout excluded the website, unrelated experiment archives and supplementary documentation; it included the integration documents required by the upstream audit. The instructions above use a full checkout.

The seeded synthetic comparison returns a certificate over two complete policies. Its selected model takes 8.3 µs in expectation at a peak of eight atoms. The released project-local Skill can also run independently of rebuilding OpenQuantum. The third-party repository and its official releases are maintained separately.
