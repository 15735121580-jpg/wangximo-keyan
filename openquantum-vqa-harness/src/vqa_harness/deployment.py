"""Independent physical deployment acceptance for a frozen complete policy.

The trusted registry seals the subject(s) before validation and authenticates
scope, acquisition order, physical gates, bounds and evidence provenance. No
policy fields, compiler scores or caller-supplied sample summaries are trusted.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import comb

from .guarded_release import finite


@dataclass(frozen=True)
class DeploymentContract:
    scope: str  # task, device/calibration epoch, answer test and metric semantics
    minimum_success: Fraction
    confidence: Fraction
    upper_caps: tuple = ()

    def __post_init__(self):
        if (not isinstance(self.scope, str) or not self.scope
                or not isinstance(self.minimum_success, Fraction) or not 0 < self.minimum_success <= 1
                or not isinstance(self.confidence, Fraction) or not 0 < self.confidence < 1):
            raise ValueError("Provide a scope, rational probability target and rational confidence level")
        if not isinstance(self.upper_caps, tuple) or any(
                type(pair) is not tuple or len(pair) != 2 or not isinstance(pair[0], str)
                or not pair[0] or not finite(pair[1]) for pair in self.upper_caps):
            raise ValueError("Provide named finite physical loss caps")
        if len(dict(self.upper_caps)) != len(self.upper_caps):
            raise ValueError("Duplicate physical cap")


@dataclass(frozen=True)
class PhysicalEvidence:
    scope: str
    subject: str  # digest of complete physical program and controller
    batch: str
    start_order: int
    end_order: int
    physical_gates_passed: bool
    # "binomial" requires independently verified i.i.d. whole-job Bernoulli trials.
    # "deterministic" requires an independently verified all-input physical proof.
    answer_method: str
    successes: int | None = None
    trials: int | None = None
    physical_upper: tuple = ()
    # Simultaneous confidence for all returned physical upper bounds. One means exact.
    bounds_confidence: Fraction | None = None


@dataclass(frozen=True)
class QualifiedSavedPolicy:
    contract: DeploymentContract
    policy: object
    subject: str
    evidence_ref: str
    batch: str
    end_order: int


@dataclass(frozen=True)
class FrozenDeployment:
    contract: DeploymentContract
    candidate: object
    subject: str
    fallback: QualifiedSavedPolicy | None
    order: int


def _tail(successes, trials, target):
    """Exact P_target[X >= successes]; inversion gives a one-sided exact LCB test."""
    if successes == 0 or target == 1:
        return Fraction(1)
    a, d = target.numerator, target.denominator
    b = d - a
    term = comb(trials, successes) * a ** successes * b ** (trials - successes)
    total = term
    for k in range(successes, trials):
        term = term * (trials - k) * a // ((k + 1) * b)
        total += term
    return Fraction(total, d ** trials)


def _assess(contract, policy, subject, reference, verifier, after=None):
    fail = []
    if not isinstance(subject, str) or not subject or verifier.subject(policy) != subject:
        return ["changed_policy"], None
    if not isinstance(reference, str) or not reference:
        return ["missing_evidence"], None
    evidence = verifier.read_evidence(contract, policy, reference)
    if not isinstance(evidence, PhysicalEvidence):
        return ["missing_evidence"], None
    if evidence.scope != contract.scope or evidence.subject != subject:
        return ["scope_or_policy_mismatch"], evidence
    if (not isinstance(evidence.batch, str) or not evidence.batch
            or type(evidence.start_order) is not int or type(evidence.end_order) is not int
            or evidence.start_order > evidence.end_order
            or (after is not None and evidence.start_order <= after)):
        return ["unverified_evidence_order"], evidence
    if evidence.physical_gates_passed is not True:
        fail.append("physical_gate_failed_or_missing")
    allowance = 1 - contract.confidence
    if contract.upper_caps:
        bounds = evidence.physical_upper
        if (not isinstance(bounds, tuple) or any(type(x) is not tuple or len(x) != 2 or
                type(x[0]) is not str or not finite(x[1]) for x in bounds) or len(dict(bounds)) != len(bounds)):
            fail.append("invalid_physical_bounds")
        else:
            values = dict(bounds)
            for name, cap in contract.upper_caps:
                if name not in values:
                    fail.append(f"missing_physical_bound:{name}")
                elif values[name] > cap:
                    fail.append(f"physical_cap_failed:{name}")
        level = evidence.bounds_confidence
        if not isinstance(level, Fraction) or not contract.confidence <= level <= 1:
            fail.append("missing_or_insufficient_bound_confidence")
        else:
            allowance -= 1 - level  # simultaneous joint control by union bound
    if evidence.answer_method == "deterministic":
        # read_evidence must verify a universal physical guarantee for the named task.
        if evidence.successes is not None or evidence.trials is not None:
            fail.append("sample_data_cannot_be_a_deterministic_proof")
    elif evidence.answer_method == "binomial":
        k, n = evidence.successes, evidence.trials
        if type(n) is not int or type(k) is not int or not 0 <= k <= n or n < 1:
            fail.append("missing_or_invalid_answer_trials")
        elif k != n:
            fail.append("observed_answer_failure")
        elif allowance <= 0 or _tail(k, n, contract.minimum_success) > allowance:
            fail.append("answer_probability_unproved")
    else:
        fail.append("missing_physical_answer_certificate")
    return fail, evidence


def audit_physical_evidence(contract, policy, evidence_ref, verifier):
    """Audit existing physical evidence without authorizing any executable."""
    reasons, _ = _assess(contract, policy, verifier.subject(policy), evidence_ref, verifier)
    return dict(status="rejected_evidence" if reasons else "meets_evidence_checks", reasons=reasons)


def qualify_saved(contract, policy, evidence_ref, verifier):
    """Check a previously validated standby against the same absolute contract."""
    subject = verifier.subject(policy)
    problems, evidence = _assess(contract, policy, subject, evidence_ref, verifier)
    saved = None if problems else QualifiedSavedPolicy(
        contract, policy, subject, evidence_ref, evidence.batch, evidence.end_order)
    return dict(status="unqualified_saved" if problems else "qualified_saved", saved=saved, reasons=problems)


def freeze_deployment(contract, candidate, verifier, fallback=None):
    """Seal the physical candidate and at most one already qualified named fallback.

    verifier.seal persists the contract, subjects and fallback evidence reference
    in its trusted ordered registry and returns a strictly ordered acquisition marker.
    """
    if not isinstance(contract, DeploymentContract):
        raise TypeError("A physical deployment contract is required")
    if fallback is not None and (not isinstance(fallback, QualifiedSavedPolicy) or fallback.contract != contract):
        raise ValueError("Standby requires prior qualification under this same contract")
    subject = verifier.subject(candidate)
    if not isinstance(subject, str) or not subject:
        raise ValueError("Supply a digest of the complete candidate and controller")
    standby = None if fallback is None else (fallback.subject, fallback.evidence_ref, fallback.batch, fallback.end_order)
    order = verifier.seal(contract, subject, standby)
    if type(order) is not int or (fallback is not None and fallback.end_order >= order):
        raise ValueError("Standby qualification must precede the candidate freeze")
    return FrozenDeployment(contract, candidate, subject, fallback, order)


def admit_deployment(frozen, evidence_ref, verifier):
    """Authorize the frozen candidate, use its previously qualified standby, or withhold."""
    standby = frozen.fallback
    registered = None if standby is None else (standby.subject, standby.evidence_ref, standby.batch, standby.end_order)
    if verifier.verify_seal(frozen.contract, frozen.subject, registered, frozen.order) is not True:
        return dict(status="not_authorized", policy=None, candidate_reasons=["unverified_freeze"])
    rejected, evidence = _assess(frozen.contract, frozen.candidate, frozen.subject,
                                 evidence_ref, verifier, after=frozen.order)
    if not rejected:
        return dict(status="qualified_candidate", policy=frozen.candidate, candidate_reasons=[])
    result = dict(status="not_authorized", policy=None, candidate_reasons=rejected)
    if standby is not None:
        reasons, prior = _assess(frozen.contract, standby.policy, standby.subject, standby.evidence_ref, verifier)
        if (prior is not None and (prior.batch != standby.batch or prior.end_order != standby.end_order
                or prior.end_order >= frozen.order or (evidence is not None and prior.batch == evidence.batch))):
            reasons.append("standby_evidence_not_independent_and_prior")
        result["standby_reasons"] = reasons
        if not reasons:
            result.update(status="qualified_standby", policy=standby.policy)
    return result


def dispatch_qualified(frozen, evidence_ref, verifier, submit):
    """Call a client-supplied device submitter only for an independently qualified policy."""
    decision = admit_deployment(frozen, evidence_ref, verifier)
    if decision["status"] in ("qualified_candidate", "qualified_standby"):
        decision["submission"] = submit(decision["policy"])
    return decision
