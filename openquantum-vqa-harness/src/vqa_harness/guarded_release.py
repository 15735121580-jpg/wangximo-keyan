"""Release a candidate against the original host using a registered, trusted verifier.

The contract and ordered branches are fixed before validation. The verifier's assess()
checks semantics and hard constraints; upper_bounds() returns sound upper bounds on
candidate-minus-host losses under that contract. Both functions must be deterministic
and independent of deployment outcomes. Unknown evidence is represented by None or
an omitted key. Arbitrary JSON provided by candidates is never a verifier.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import math


def finite(value):
    return type(value) in (int, Fraction) or (type(value) is float and math.isfinite(value))


@dataclass(frozen=True)
class ReleaseContract:
    model: str
    losses: tuple
    caps: tuple = ()
    reward: str | None = None

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("Register a model name")
        if not isinstance(self.losses, tuple) or not self.losses or any(
                type(k) is not str or not k for k in self.losses) or len(set(self.losses)) != len(self.losses):
            raise ValueError("Register a nonempty ordered tuple of unique loss names")
        if self.reward is not None:
            if type(self.reward) is not str or not self.reward or self.reward in self.losses:
                raise ValueError("Name the maximized reward separately from ordinary losses")
            reward_loss = "-" + self.reward
            if reward_loss not in self.losses:
                object.__setattr__(self, "losses", (*self.losses, reward_loss))
        if not isinstance(self.caps, tuple) or any(type(pair) is not tuple or len(pair) != 2
                or type(pair[0]) is not str or not pair[0] or not finite(pair[1]) for pair in self.caps):
            raise ValueError("Register caps as (resource, maximum) tuples")
        if len(dict(self.caps)) != len(self.caps):
            raise ValueError("Duplicate resource cap")


@dataclass(frozen=True)
class Assessment:
    # "infeasible" requires affirmative evidence; missing/invalid cannot authorize recovery.
    status: str
    evidence: object = None


def checked(assessment):
    if isinstance(assessment, Assessment) and assessment.status in (
            "feasible", "infeasible", "missing", "invalid"):
        return assessment
    return Assessment("invalid")


def audit_bounds(losses, bounds):
    if bounds is None:
        return "missing_bounds", {}
    if not isinstance(bounds, Mapping):
        return "invalid_bounds", {}
    supplied = {key: bounds.get(key) for key in losses}
    if any(v is not None and not finite(v) for v in supplied.values()):
        return "invalid_bounds", {}
    if any(v is None for v in supplied.values()):
        return "missing_bounds", supplied
    if max(supplied.values()) > 0:
        # A positive upper bound also covers unresolved signs.
        return "non_regression_unproved", supplied
    if min(supplied.values()) == 0:
        return "no_certified_gain", supplied
    return "eligible", supplied


_HOST_ONLY = object()


def guarded_release(contract, host, candidates, verifier, *, incumbent=_HOST_ONLY):
    """Return policy and audit. Equal bounds use registered branch order.

    Among eligible branches the registered order of loss names is the
    lexicographic priority on certified upper differences from B. With a
    protected incumbent, require every difference from both references to
    be nonpositive and at least one difference from the incumbent to be negative.
    Host-only recovery selects the first certified feasible branch. The returned policy is the original
    object for whichever branch wins. If the host cannot be assessed, no
    release occurs. The API never invokes the policy or reads its outcomes.
    """
    if not isinstance(contract, ReleaseContract) or verifier.model != contract.model:
        raise ValueError("Contract and verifier model must match")
    original = checked(verifier.assess(contract, host))
    records = []
    result = dict(host_status=original.status, branches=records, selected=None, policy=None)
    if original.status in ("invalid", "missing"):
        return dict(result, status="unresolved_host")
    dual = incumbent is not _HOST_ONLY
    distinct = dual and incumbent is not host
    if dual:
        result["comparison"] = "host_and_direct_incumbent"
        if original.status != "feasible":
            return dict(result, status="infeasible_host")
        if incumbent is None:
            return dict(result, status="unresolved_incumbent")
        current = original if not distinct else checked(verifier.assess(contract, incumbent))
        result["incumbent_status"] = current.status
        if current.status != "feasible":
            return dict(result, status="unresolved_incumbent")
        if distinct:
            # Recheck delivery under this exact contract even for a previously saved receipt.
            approval, values = audit_bounds(contract.losses, verifier.upper_bounds(contract, original, current))
            result["incumbent_vs_host"] = values
            if approval not in ("eligible", "no_certified_gain"):
                return dict(result, status="unresolved_incumbent")
    winner = None
    best = None
    for i, proposal in enumerate(candidates):
        verified = checked(verifier.assess(contract, proposal))
        record = dict(index=i, feasibility=verified.status)
        records.append(record)
        if verified.status != "feasible":
            record["decision"] = verified.status
            continue
        if original.status == "infeasible":
            record["decision"] = "eligible_recovery"
            if winner is None:
                winner = (i, proposal)
            continue
        bounds = verifier.upper_bounds(contract, original, verified)
        decision, values = audit_bounds(contract.losses, bounds)
        record.update(decision=decision, upper_differences=values)
        if dual:
            record["upper_vs_host"] = values
            if distinct and decision in ("eligible", "no_certified_gain"):
                # Historical verifiers may be sound only against the original host.
                # Generic pairwise support requires a separate, explicit certificate.
                bounder = getattr(verifier, "upper_bounds_against", None)
                if proposal is incumbent:
                    bounds = {name: 0 for name in contract.losses}
                else:
                    bounds = bounder(contract, current, verified) if callable(bounder) else None
                decision, against = audit_bounds(contract.losses, bounds)
                record.update(decision=decision, upper_vs_incumbent=against)
            elif not distinct:
                record["upper_vs_incumbent"] = values
        if decision == "eligible":
            key = tuple(values[k] for k in contract.losses)
            if best is None or key < best:
                best, winner = key, (i, proposal)
    if winner is not None:
        return dict(result, status="released" if original.status == "feasible" else "recovery",
                    selected=winner[0], policy=winner[1])
    if distinct:
        return dict(result, status="retained_incumbent", policy=incumbent)
    if original.status == "feasible":
        return dict(result, status="retained_host", policy=host)
    return dict(result, status="no_certified_recovery")
