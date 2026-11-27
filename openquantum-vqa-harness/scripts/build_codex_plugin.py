"""Build the wheel and copy the runtime assets into the Codex plugin."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins/openquantum-vqa-harness"
SKILL = PLUGIN / "skills/openquantum-vqa-harness"


def main():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    with tempfile.TemporaryDirectory(prefix="openquantum-wheel-") as build:
        subprocess.run(
            ["uv", "build", "--wheel", str(ROOT), "--out-dir", build], check=True,
        )
        wheel = Path(build) / f"openquantum_vqa_harness-{project['version']}-py3-none-any.whl"
        packages = PLUGIN / "packages"
        packages.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wheel, packages / wheel.name)
        for previous in packages.glob("openquantum_vqa_harness-*.whl"):
            if previous.name != wheel.name:
                previous.unlink()
    shutil.copytree(ROOT / "skills/vqa-joint-compilation", SKILL, dirs_exist_ok=True)
    shutil.copyfile(ROOT / "examples/three-spin-vqe.json", SKILL / "assets/three-spin-vqe.json")
    print(PLUGIN)


if __name__ == "__main__":
    main()
