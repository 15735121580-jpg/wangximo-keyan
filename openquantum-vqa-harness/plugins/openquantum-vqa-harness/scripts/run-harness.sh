#!/bin/sh
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
set -- "$root"/packages/openquantum_vqa_harness-*.whl
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    printf '%s\n' 'Expected one bundled OpenQuantum VQA Harness wheel.' >&2
    exit 1
fi
if command -v uv >/dev/null 2>&1; then
    uv_command=uv
elif [ -x "$HOME/.local/bin/uv" ]; then
    uv_command="$HOME/.local/bin/uv"
elif [ -x /opt/homebrew/bin/uv ]; then
    uv_command=/opt/homebrew/bin/uv
else
    printf '%s\n' 'Install uv (https://docs.astral.sh/uv/) and reopen Codex.' >&2
    exit 1
fi
exec "$uv_command" --no-config tool run --from "$1" --python 3.12 vqa-joint-mcp
