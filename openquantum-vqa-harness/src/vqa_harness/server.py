"""MCP tools shared by Codex and Claude Code. Scientific execution uses OpenQuantum."""
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from . import __version__
from .runtime import execute, openquantum_sdk
from .native_policy import compile_native_policy

server = FastMCP("OpenQuantum VQA Joint Harness")


@server.tool()
def compile_joint_job(task: dict, platform: str, tolerance: str | None = None) -> dict:
    """Return a certified optimized policy on a locally registered platform.

    tolerance is an absolute rational in the adapter's objective unit. An open
    result reports bounds and its incumbent separately, with no released policy.
    """
    from .certified_entry import compile_registered
    return compile_registered(task, platform, tolerance)


@server.tool()
def compile_native_host_policy(job: dict) -> dict:
    """Emit one bound native policy from a host's CZ/rotation or two-wire RZZ interface.

    Requires explicit hardware capabilities, basis, and logical or native source per
    binding. Returns native operations and the relevant readout-map or slot changes.
    """
    return compile_native_policy(job)


@server.tool()
def harness_status() -> dict:
    """Return the installed harness version and prepared OpenQuantum SDK metadata."""
    return {"version": __version__, "runtime": openquantum_sdk()}


@server.tool()
def compile_vqa_job(job: dict, output_directory: str, backend: str = "openquantum") -> dict:
    """Compile and execute a complete finite-grid VQA in a declared device model.

    Supply task gates/product input/Pauli terms (q0-first), device topology/native
    basis/timings/fidelities, and vqa parameter_grid/shots/terminal_shots/seed.
    Returns actual training and fresh-terminal answers, native programs and
    cost/semantic certificates. The output directory must be new. The default
    uses OpenQuantum's prepared local SDK. Model noise and all sample budgets
    are specified by the caller. This tool performs local simulation.
    """
    if not Path(output_directory).is_absolute():
        raise ValueError("output_directory must be absolute")
    receipt = execute(job, output_directory, backend)
    return {
        "status": receipt["status"],
        "answer": receipt["answer"],
        "arms": {name: {"answer": data["answer"], "resources": data["resources"]}
                 for name, data in receipt["arms"].items()},
        "verification": {key: value for key, value in receipt["verification"].items()
                         if key != "checks"},
        "reference": {key: value for key, value in receipt["reference"].items()
                      if key != "ideal_grid_energies"},
        "comparison": receipt["comparison"],
        "scope": receipt["scope"],
        "artifacts": receipt["artifacts"],
    }


def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
