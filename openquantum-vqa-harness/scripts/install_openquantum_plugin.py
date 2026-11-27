"""Merge the packaged VQA harness into an existing OpenQuantum plugin source."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile


REPO = Path(__file__).resolve().parents[1]
SKILL = "openquantum-vqa-harness"
SERVER = "vqa-compiler"
COMPONENT = Path("harness") / SERVER
LAUNCHER = Path("scripts/run-harness.sh")
RECEIPT = "integration.json"


def read_json(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def plugin_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a plugin-relative path, got {relative}")
    resolved = (root / path).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Path must stay inside the plugin: {relative}")
    if resolved != root / path:
        raise ValueError(f"Symbolic links are unsupported at managed paths: {relative}")
    return resolved


def manifest(root, name):
    data = read_json(plugin_path(root, ".codex-plugin/plugin.json"))
    if data.get("name") != name:
        raise ValueError(f"Expected plugin {name!r} in {root}")
    return data


def fingerprint(root):
    """Fingerprint names, object types and file contents without following symlinks."""
    items = []
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            items.append((name, "link", str(path.readlink())))
        elif path.is_file():
            mode = stat.S_IMODE(path.stat().st_mode)
            items.append((name, "file", mode, hashlib.sha256(path.read_bytes()).hexdigest()))
        elif path.is_dir():
            items.append((name, "dir"))
        else:
            raise ValueError(f"Unsupported filesystem object: {path}")
    return hashlib.sha256(json.dumps(items, ensure_ascii=True).encode()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def replace_plugin(target, replacement, snapshot):
    """Preserve the source even if publishing the staged plugin is interrupted."""
    target.rename(snapshot)
    try:
        replacement.rename(target)
    except BaseException:
        snapshot.rename(target)
        raise


def install(target, bundle, dry_run):
    before = fingerprint(target)
    definition = manifest(target, "openquantum")
    bundle_manifest = manifest(bundle, SKILL)
    skills = plugin_path(target, definition.get("skills", "./skills"))
    mcp = plugin_path(target, definition.get("mcpServers", "./mcp.json"))
    source_skill = plugin_path(bundle, Path(bundle_manifest.get("skills", "./skills")) / SKILL)
    launcher = plugin_path(bundle, LAUNCHER)
    packages = plugin_path(bundle, "packages")
    incoming = read_json(plugin_path(bundle, bundle_manifest.get("mcpServers", "./mcp.json")))
    server = incoming["mcpServers"][SERVER]
    if server.get("command") != "sh" or server.get("args", []) != ["${PLUGIN_ROOT}/" + LAUNCHER.as_posix()]:
        raise ValueError("The companion plugin must use its packaged shell launcher")
    if not (skills.is_dir() and (source_skill / "SKILL.md").is_file() and
            launcher.is_file()):
        raise ValueError("Missing OpenQuantum skill directory, harness skill or launcher")
    wheels = list(packages.glob("openquantum_vqa_harness-*.whl"))
    if len(wheels) != 1 or not wheels[0].is_file() or wheels[0].is_symlink():
        raise ValueError("The companion plugin must contain exactly one harness wheel in packages/")
    component = plugin_path(target, COMPONENT)
    destination_skill = plugin_path(target, skills.relative_to(target) / SKILL)
    config = read_json(mcp)
    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        raise ValueError(f"No mcpServers object in {mcp}")
    server["args"] = ["${PLUGIN_ROOT}/" + (COMPONENT / LAUNCHER).as_posix()]

    matches = (
        servers.get(SERVER) == server
        and destination_skill.is_dir()
        and fingerprint(destination_skill) == fingerprint(source_skill)
        and (component / LAUNCHER).is_file()
        and (component / LAUNCHER).read_bytes() == launcher.read_bytes()
        and (component / "packages").is_dir()
        and fingerprint(component / "packages") == fingerprint(packages)
    )
    if matches:
        print(f"Harness already integrated in {target}")
        return
    if SERVER in servers or destination_skill.exists() or component.exists():
        raise ValueError("An existing harness entry would be overwritten. Restore the previous "
                         "installation before integrating a different bundle.")
    if dry_run:
        print(f"Can integrate skill, launcher, {wheels[0].name} and MCP server in {target}")
        return

    backup = Path(tempfile.mkdtemp(prefix=".openquantum-before-vqa-", dir=target.parent))
    original = backup / "original"
    staged = backup / "staged"
    try:
        shutil.copytree(target, staged, symlinks=True)
        staged_component = staged / COMPONENT
        (staged_component / "scripts").mkdir(parents=True)
        shutil.copy2(launcher, staged_component / LAUNCHER)
        shutil.copytree(packages, staged_component / "packages", symlinks=True)
        shutil.copytree(source_skill, staged / destination_skill.relative_to(target), symlinks=True)
        servers[SERVER] = server
        write_json(staged / mcp.relative_to(target), config)
        write_json(backup / RECEIPT, {
            "target": str(target), "before": before, "integrated": fingerprint(staged)
        })
        if fingerprint(target) != before:
            raise RuntimeError("The OpenQuantum plugin changed during integration. Please rerun.")
        replace_plugin(target, staged, original)
    except BaseException:
        # If restoration itself failed, retain the original tree for manual recovery.
        if not original.exists():
            shutil.rmtree(backup)
        raise
    print(f"Integrated {wheels[0].name} and its Codex skill in {target}")
    print(f"Rollback directory: {backup}")


def rollback(target, backup, dry_run):
    receipt = read_json(backup / RECEIPT)
    original = backup / "original"
    integrated = backup / "integrated"
    if receipt.get("target") != str(target) or not original.is_dir() or integrated.exists():
        raise ValueError(f"No matching rollback snapshot in {backup}")
    if fingerprint(target) != receipt["integrated"] or fingerprint(original) != receipt["before"]:
        raise ValueError("Plugin or snapshot contents changed after integration. Rollback stopped.")
    if dry_run:
        print(f"Can restore original OpenQuantum plugin in {target}")
        return
    replace_plugin(target, original, integrated)
    print(f"Restored original OpenQuantum plugin in {target}")
    print(f"Previous integrated copy: {integrated}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="OpenQuantum plugin source directory")
    parser.add_argument("--bundle", type=Path, default=REPO / "plugins" / SKILL,
                        help="directory containing the packaged harness Codex plugin")
    parser.add_argument("--dry-run", action="store_true", help="check compatibility without writing")
    parser.add_argument("--rollback", type=Path, metavar="BACKUP",
                        help="restore the OpenQuantum snapshot from a previous integration")
    args = parser.parse_args()
    try:
        target = args.target.expanduser().resolve(strict=True)
        if args.rollback is None:
            install(target, args.bundle.expanduser().resolve(strict=True), args.dry_run)
        else:
            rollback(target, args.rollback.expanduser().resolve(strict=True), args.dry_run)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"OpenQuantum integration: {error}\n")


if __name__ == "__main__":
    main()
