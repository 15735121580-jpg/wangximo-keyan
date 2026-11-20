# Discovering and contributing the Skill

For a project-local Codex or OpenQuantum workspace, copy the complete `vqa-joint-compilation` folder to `.agents/skills/vqa-joint-compilation` at the workspace root. Open a fresh conversation in that workspace and explicitly invoke `$vqa-joint-compilation`. The scripts and assets travel with the folder. The OpenQuantum plugin can remain installed at its current version. The local overlay does not register a new Tool, MCP server or OpenQuantum acceptance profile.

```sh
mkdir -p .agents/skills
cp -R /path/to/vqa-joint-compilation .agents/skills/vqa-joint-compilation
```

To build this Skill as part of OpenQuantum's own Codex plugin, follow the [upstream integration recipe and patch](https://github.com/15735121580-jpg/wangximo-keyan/tree/main/openquantum-vqa-joint/integration). It installs the source in the upstream capability registry at level L0 and teaches the release builder to ship the Python entry point, model contracts, certified algorithms and native catalogue beside the generated host Skill. The host runs these scientific programs with its local code tool. The recipe also checks the packaged capability and replays a two-round job from the actual release archive.

Keep time, sample, noise and architecture evidence in the JSON/SDK inputs. Raw hardware data, credentials and any current backend authorization are supplied independently of the Skill. The bundled source trees are the maintained VQA Joint 0.1.0a5 native-catalogue CLI and the certified-policy research SDK. The native catalogue retains the ZAC hardware specification and its upstream licence.
