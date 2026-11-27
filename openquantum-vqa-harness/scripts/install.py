"""Register the local harness as an MCP server and install its host Skill."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys


def install(host, python, root):
    name = "vqa-joint-harness"
    commands = {
        "codex": ["codex", "mcp", "add", name, "--", str(python), "-m", "vqa_harness.server"],
        "claude": ["claude", "mcp", "add", "--scope", "user", name, "--", str(python),
                   "-m", "vqa_harness.server"],
    }
    current = subprocess.run([host, "mcp", "get", name], capture_output=True, text=True)
    already_installed = current.returncode == 0 and str(python) in current.stdout and "vqa_harness.server" in current.stdout
    # Backups remain private beside the existing host configuration.
    config = Path.home() / (".codex/config.toml" if host == "codex" else ".claude.json")
    if config.is_file() and not already_installed:
        backup = config.with_name(config.name + ".before-vqa-" + datetime.now().strftime("%Y%m%d%H%M%S"))
        shutil.copy2(config, backup)
        backup.chmod(0o600)
    if not already_installed:
        if current.returncode == 0 and host == "claude":
            subprocess.run(["claude", "mcp", "remove", name, "--scope", "user"], check=True)
        subprocess.run(commands[host], check=True)
    base = Path.home() / (".agents/skills" if host == "codex" else ".claude/skills")
    destination = base / "openquantum-vqa-harness"
    destination.mkdir(parents=True, exist_ok=True)
    text = (root / "skills/vqa-joint-compilation/SKILL.md").read_text()
    text += "\nInstalled source checkout " + str(root) + "\n"
    text += "CLI fallback " + str(python.parent / "vqa-joint") + "\n"
    (destination / "SKILL.md").write_text(text)
    print("Installed Skill", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["codex", "claude", "both"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    python = Path(sys.executable).absolute()
    for host in (["codex", "claude"] if args.host == "both" else [args.host]):
        install(host, python, root)


if __name__ == "__main__":
    main()
