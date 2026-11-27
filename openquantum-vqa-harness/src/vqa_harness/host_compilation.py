"""Apply one rewrite or a bounded proposal list to the same registered host job."""
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from .guarded_release import ReleaseContract, guarded_release


def _names(names):
    return (type(names) is tuple and all(type(s) is str and s for s in names)
            and len(names) == len(set(names)))


@dataclass(frozen=True)
class HostJob:
    """The caller freezes the task, hardware, host policy/controller and exposed interfaces."""
    task: object
    hardware: object
    host: object
    stages: tuple[str, ...]
    contract: ReleaseContract
    execution_budget: object = None  # Explicit shot/time/quota scope, if it is not already fixed in task.

    def __post_init__(self):
        if self.task is None or self.hardware is None or self.host is None:
            raise ValueError("Provide task, hardware and an original host policy")
        if not _names(self.stages) or not isinstance(self.contract, ReleaseContract):
            raise ValueError("Provide unique exposed stages and a release contract")


@dataclass(frozen=True)
class Branch:
    """An adapter-specific frozen recipe and the exposed stages it may change."""
    name: str
    stages: tuple[str, ...]

    def __post_init__(self):
        if type(self.name) is not str or not self.name or not self.stages or not _names(self.stages):
            raise ValueError("Provide a named branch with unique target stages")


@dataclass(frozen=True)
class DirectIncumbent:
    """The direct result bound to its original immutable task, device, budget and metrics."""
    job: HostJob
    policy: object
    key: object
    steps: tuple = ()


def _snapshot(value):
    """Freeze standard structured client inputs without importing a host SDK."""
    if value is None or type(value) in (bool, int, float, str, bytes, Fraction):
        return (type(value), value)
    if type(value) in (list, tuple):
        return (type(value), tuple(_snapshot(v) for v in value))
    if type(value) in (set, frozenset):
        return (type(value), frozenset(_snapshot(v) for v in value))
    if type(value) is dict:
        return (dict, frozenset((_snapshot(k), _snapshot(v)) for k, v in value.items()))
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((f.name, _snapshot(getattr(value, f.name))) for f in fields(value)))
    raise TypeError("Opaque SDK state needs a trusted incumbent_key(job, policy)")


def _binding(job, policy, verifier):
    custom = getattr(verifier, "incumbent_key", None)
    if callable(custom):
        digest = custom(job, policy)
        if isinstance(digest, (str, bytes)) and digest:
            return (job.contract, digest)
        return None
    try:
        return _snapshot((job, policy))
    except (TypeError, RecursionError):
        return None


def direct_receipt(job, result, verifier, steps=()):
    if result["status"] not in ("released", "retained_host"):
        return None
    policy = result["policy"]
    key = _binding(job, policy, verifier)
    return DirectIncumbent(job, policy, key, tuple(steps)) if key is not None else None


def incumbent_policy(job, incumbent, verifier):
    if isinstance(incumbent, dict):
        incumbent = incumbent.get("incumbent")
    if not isinstance(incumbent, DirectIncumbent):
        return None
    key = _binding(job, incumbent.policy, verifier)
    if key is None or incumbent.key != key:
        return None
    return incumbent


def _compile(job, branches, budget, adapter, *, direct=False, incumbent=None, legacy_host_only=False):
    if not isinstance(job, HostJob) or type(branches) is not tuple:
        raise ValueError("Register a HostJob and an ordered tuple of branches")
    if type(budget) is not int or budget < 0:
        raise ValueError("Budget must be a nonnegative integer")
    if any(not isinstance(b, Branch) or not set(b.stages).issubset(job.stages) for b in branches):
        raise ValueError("A branch addresses an unregistered stage")
    if len({b.name for b in branches}) != len(branches):
        raise ValueError("Branch names must be unique")
    scheduled = branches[:budget]
    verifier = adapter.verifier(job)
    candidates = (adapter.construct(job, branch) for branch in scheduled)
    if direct or legacy_host_only:
        result = guarded_release(job.contract, job.host, candidates, verifier)
    else:
        if incumbent is None:
            recipe = getattr(adapter, "direct_branch", None)
            if callable(recipe):
                incumbent = compile_direct(job, recipe(job), adapter)
        ticket = incumbent_policy(job, incumbent, verifier)
        result = guarded_release(job.contract, job.host, candidates, verifier,
                                 incumbent=None if ticket is None else ticket.policy)
    for record, branch in zip(result["branches"], scheduled):
        record.update(recipe=branch.name, stages=branch.stages)
    if direct:
        result["incumbent"] = direct_receipt(job, result, verifier)
    return dict(result, registered=len(branches), budget=budget, evaluated=len(result["branches"]))


def compile_bounded(job, branches, budget, adapter, *, incumbent=None, legacy_host_only=False):
    """A2. Protect feasible B and its delivered A1 under the same immutable HostJob.

    Pass a compile_direct receipt or let adapter.direct_branch(job) prescribe A1.
    An unknown incumbent cannot authorize a program. The branch budget counts
    new search evaluations and is independent of a physical execution budget.
    Set legacy_host_only=True solely to replay the v0.2 single-host comparator.
    """
    if type(legacy_host_only) is not bool or (legacy_host_only and incumbent is not None):
        raise ValueError("Choose the protected incumbent or explicit host-only replay")
    return _compile(job, branches, budget, adapter, incumbent=incumbent, legacy_host_only=legacy_host_only)


def compile_direct(job, branch, adapter):
    """A1. Check one proposal against B and bind its delivered policy to the same job."""
    return _compile(job, (branch,), 1, adapter, direct=True)
