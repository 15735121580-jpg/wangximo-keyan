from __future__ import annotations
"""Typed study contract for comparing two complete VQA executions."""
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def __post_init__(self):
        if not (isfinite(self.low) and isfinite(self.high) and self.low <= self.high):
            raise ValueError("Expected an ordered finite interval")

    def probability(self):
        if self.low < 0 or self.high > 1:
            raise ValueError("Probability bounds must lie in [0,1]")
        return self


@dataclass(frozen=True)
class Platform:
    """Shared operating assumptions. round_logical_error is a reference only."""
    device: str
    calibration: str
    code_decoder_round: str
    gate_fidelities: tuple[tuple[str, Interval], ...]
    round_logical_error: Interval | None
    channel_model: str | None
    control: str = 'digital'

    @classmethod
    def neutral_atom_study(cls, device, *, calibration, code_decoder_round,
                           channel_model=None, reference_round_error=None):
        return cls(device, calibration, code_decoder_round, DIGITAL_GATE_INPUTS,
                   reference_round_error, channel_model)

    def __post_init__(self):
        if self.control not in ('digital','analog'):
            raise ValueError("Specify digital gates or analog pulse control")
        if not (self.device and self.calibration and self.code_decoder_round):
            raise ValueError("The source and the round convention are required")
        names = set()
        for gate, interval in self.gate_fidelities:
            if not gate or gate in names:
                raise ValueError("Gate types must be explicit and unique")
            names.add(gate)
            interval.probability()
        if self.round_logical_error is not None:
            self.round_logical_error.probability()


# Average fidelities of a physical gate acting on its stated ideal target.
# Readout, loss, transport and native multiqubit pulses have separate models.
DIGITAL_GATE_INPUTS = (('1q_average', Interval(0.9999, 0.9999)),
                       ('2q_average', Interval(0.999, 0.999)))

RESOURCE_OBJECTIVES = ('depth', 'peak_physical_qubits', 'physical_gates',
                       'circuit_process_infidelity', 'execution_seconds')
OBJECTIVES = (*RESOURCE_OBJECTIVES, 'logical_error')
PULSE_OBJECTIVES = ('pulse_depth','peak_physical_qubits','physical_pulses','circuit_process_infidelity','execution_seconds')
LARGE_GAIN_THRESHOLD = Decimal('0.35')


@dataclass(frozen=True)
class LogicalEvidence:
    """One arm's decoded logical failure under a specified experiment."""
    endpoint: str
    protocol: str
    schedule: str
    source: str
    bound: Interval

    def __post_init__(self):
        if self.endpoint not in ('per_qec_round', 'accepted_magic_state',
                                 'whole_encoded_job'):
            raise ValueError('Specify the logical failure event and denominator')
        if not all((self.protocol, self.schedule, self.source)):
            raise ValueError('Logical error needs a matched test, schedule and evidence source')
        self.bound.probability()


@dataclass(frozen=True)
class Assessment:
    # Same task includes dataset instance, solver, stop rule, output score,
    # confidence/precision, timing convention, and branch aggregation rules.
    task_contract: str
    platform: Platform
    success_probability: Interval | None
    objectives: Mapping[str, Interval | None]
    decoded: LogicalEvidence | None = None
    encoding: str = 'encoded'

    @property
    def resource_objectives(self):
        return PULSE_OBJECTIVES if self.platform.control == 'analog' else RESOURCE_OBJECTIVES

    def __post_init__(self):
        if self.encoding not in ('encoded', 'unencoded'):
            raise ValueError('Declare whether the physical workload uses an error-correcting code')
        if self.encoding == 'unencoded' and self.decoded is not None:
            raise ValueError('An unencoded job has no decoded logical error event')
        # Legacy records retain their five measured slots; their sixth is unknown.
        if set(self.objectives) not in (set(self.resource_objectives), set(self.resource_objectives)|{'logical_error'}):
            raise ValueError("Specify each resource, with unknowns marked None")
        values = dict(self.objectives)
        values.setdefault('logical_error', self.decoded.bound if self.decoded else None)
        if values['logical_error'] != (self.decoded.bound if self.decoded else None):
            raise ValueError("A logical-error estimate must match evidence for this schedule")
        object.__setattr__(self, 'objectives', MappingProxyType(values))
        if self.success_probability is not None:
            self.success_probability.probability()
        for name, bound in self.objectives.items():
            if bound is None:
                continue
            if bound.low < 0:
                raise ValueError("Negative resource use")
            if name in ('circuit_process_infidelity', 'logical_error'):
                bound.probability()


def compare(candidate, reference, required_success):
    """Compare on a common simultaneous coverage event or exact bounds.

    This routine does not construct confidence intervals or turn unknowns into
    hardware evidence. An inconclusive result carries the per-coordinate data.
    """
    if not 0 <= required_success <= 1:
        raise ValueError("Invalid task threshold")
    if (candidate.task_contract != reference.task_contract or candidate.platform != reference.platform
            or candidate.encoding != reference.encoding):
        raise ValueError("A paired comparison must use the same task and calibration")
    if candidate.decoded and reference.decoded:
        left, right = candidate.decoded, reference.decoded
        if (left.endpoint, left.protocol) != (right.endpoint, right.protocol):
            raise ValueError("Match logical tests and denominators across arms")
    eligible = all(r.success_probability is not None and
                   r.success_probability.low >= required_success
                   for r in (candidate, reference))
    strict, unresolved, worse = [], [], []
    success_a, success_b = candidate.success_probability, reference.success_probability
    if success_a is None or success_b is None:
        unresolved.append('task_success')
    elif success_a.low > success_b.high:
        strict.append('task_success')
    elif success_a.high < success_b.low:
        worse.append('task_success')
    elif success_a.low < success_b.high:
        unresolved.append('task_success')
    applicable = (*candidate.resource_objectives, 'logical_error') if candidate.encoding=='encoded' else candidate.resource_objectives
    for key in applicable:
        a, b = candidate.objectives[key], reference.objectives[key]
        if a is None or b is None:
            unresolved.append(key)
        elif a.high < b.low:
            strict.append(key)
        elif a.low > b.high:
            worse.append(key)
        elif a.high > b.low:
            unresolved.append(key)
    physical_model = bool(candidate.platform.channel_model and (candidate.platform.control=='analog'
                          or candidate.platform.gate_fidelities==DIGITAL_GATE_INPUTS))
    ft_evidence = bool(physical_model and candidate.decoded and reference.decoded)
    whole_task_evidence = physical_model if candidate.encoding == 'unencoded' else (
        ft_evidence and candidate.decoded.endpoint != 'accepted_magic_state')
    return dict(both_pass_task_threshold=eligible, matched_logical_evidence=ft_evidence,
                logical_endpoint=candidate.decoded.endpoint if candidate.decoded else None,
                encoding=candidate.encoding, not_applicable=sorted((set(candidate.resource_objectives)|{'logical_error'})-set(applicable)),
                strict_improvements=strict, unresolved=unresolved, worse=worse,
                certified_tie=bool(eligible and whole_task_evidence and not strict and not unresolved and not worse),
                certified_pareto=bool(eligible and whole_task_evidence and strict
                                      and not unresolved and not worse))


def large_gain(candidate, reference, primary, required_success):
    """Apply the inclusive threshold to a predeclared whole-job cost.

    With simultaneous bounds, the worst admissible relative gain uses the
    candidate's upper cost and the reference's lower cost. A zero reference
    cannot certify a positive relative saving.
    """
    if primary not in (*candidate.resource_objectives,'logical_error'):
        raise ValueError('Specify a predeclared whole-job objective')
    result = compare(candidate, reference, required_success)
    base = reference.objectives[primary]
    after = candidate.objectives[primary]
    gain = None
    if base is not None and base.low > 0 and after is not None:
        gain = 1 - Decimal(str(after.high)) / Decimal(str(base.low))
    qualifies = result['certified_pareto'] and gain is not None and gain >= LARGE_GAIN_THRESHOLD
    return dict(**result, primary=primary, threshold=float(LARGE_GAIN_THRESHOLD),
                gain_lower=float(gain) if gain is not None else None,
                qualifies_large_gain=bool(qualifies))
