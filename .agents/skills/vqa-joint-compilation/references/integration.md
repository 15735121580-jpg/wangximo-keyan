# Discovering and contributing the Skill

For a project-local Codex or OpenQuantum workspace, copy the complete `vqa-joint-compilation` folder to `.agents/skills/vqa-joint-compilation` at the workspace root. Open a fresh conversation in that workspace and explicitly invoke `$vqa-joint-compilation`. The scripts and assets travel with the folder. The OpenQuantum plugin can remain installed at its current version. The local overlay does not register a new Tool, MCP server or OpenQuantum acceptance profile.

```sh
mkdir -p .agents/skills
cp -R /path/to/vqa-joint-compilation .agents/skills/vqa-joint-compilation
```

For a contribution to the OpenQuantum repository, use the same location for the shared source. Add this new list entry under `packages` in `.agents/capability-packages.yml`.

```yaml
  - id: vqa-joint-compilation
    level: L0
```

Stage the entire skill folder and the manifest change with `git add .agents/skills/vqa-joint-compilation .agents/capability-packages.yml`. Run `npm run capability:conformance` in the OpenQuantum checkout, then verify discovery on the actual target adapter. Use native OpenQuantum registration if adding an actual tool later. The [upstream contributing guide](https://github.com/xi-zhao/OpenQuantum/blob/main/CONTRIBUTING.md#增加量子-skill) documents this separation.

For a bundled Codex distribution, the maintainer builds with `node scripts/build-codex-plugin.mjs` and checks the installed plugin's skill discovery. The current projection generates `skills/<id>/SKILL.md` and places tracked source files under `server/.agents/skills/<id>/`. A maintainer should expose that local source path from the projected Skill or explicitly include the runnable resources alongside the projected entry. The standalone project-local copy already uses sibling paths. See the [Codex packaging script](https://github.com/xi-zhao/OpenQuantum/blob/main/scripts/build-codex-plugin.mjs) and [Skill projection](https://github.com/xi-zhao/OpenQuantum/blob/main/scripts/build-chatgpt.mjs).

Keep time, sample, noise and architecture evidence in the JSON/SDK inputs. Raw hardware data, credentials and any current backend authorization are supplied independently of the Skill. The bundled source trees are the maintained VQA Joint 0.1.0a5 native-catalogue CLI and the certified-policy research SDK. The native catalogue retains the ZAC hardware specification and its upstream licence.
