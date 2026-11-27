"""Local JSON transport for installed, operator-registered physical adapters."""
from fractions import Fraction
from importlib import import_module
import json
import os

from .optimality import compile_joint


def compile_registered(task, platform, tolerance=None):
    """Use a trusted named adapter; task input cannot specify executable imports."""
    configured = json.loads(os.environ.get("VQA_JOINT_ADAPTERS", "{}"))
    if not isinstance(configured, dict) or platform not in configured:
        raise ValueError(f"Register the local platform adapter {platform!r} in VQA_JOINT_ADAPTERS")
    module, obj = configured[platform].split(":")
    adapter = getattr(import_module(module), obj)()
    if tolerance is not None:
        if type(tolerance) not in (int, str):
            raise ValueError("Use an integer or exact rational string for the absolute tolerance")
        try:
            tolerance = Fraction(tolerance)
        except (ZeroDivisionError, ValueError) as exc:
            raise ValueError("Use an integer or exact rational string for the absolute tolerance") from exc
    receipt = compile_joint(task, adapter, tolerance=tolerance)
    for key in ("policy", "incumbent"):
        if receipt.get(key) is not None and hasattr(adapter, "export"):
            receipt[key] = adapter.export(receipt[key])
    for key in ("lower", "upper", "gap", "tolerance"):
        if receipt[key] is not None:
            receipt[key] = str(receipt[key])
    return receipt
