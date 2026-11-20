"""Extract and replay a host-specific program for the stages of a closed-loop job.

A stage is a call site selected by the application's existing feedback controller.
Each candidate fixes its instrument, native profile, compilation rule and wire port
across all supplied bindings of that site. The caller's evaluator owns the whole
feedback loop, including selection of parameters, shots, stopping and branch costs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from math import isfinite, prod
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

from vqa_joint.contract import Assessment, OBJECTIVES, PULSE_OBJECTIVES, compare


class PolicyRefusal(ValueError):
    """A call or controller failed the declared guard."""


@dataclass(frozen=True)
class Binding:
    n: int
    ops: tuple
    calls: tuple


def _shape(ops, n):
    if not isinstance(n, Integral) or n < 1:
        raise ValueError("Invalid logical register width")
    body = []
    for name, wires, angle in ops:
        wires = tuple(wires)
        if not wires or len(set(wires)) != len(wires) or any(
            not isinstance(q, Integral) or not 0 <= q < n for q in wires
        ):
            raise ValueError("Invalid logical wires")
        if angle is not None and (not isinstance(angle, Real) or not isfinite(angle)):
            raise ValueError("Expected a finite parameter binding")
        body.append((name.lower(), tuple(map(int, wires)), angle is not None))
    return int(n), tuple(body)


@dataclass(frozen=True, order=True)
class Strategy:
    family: str
    variant: str
    profile: str
    implementation: str
    wire_order: tuple

    @classmethod
    def of(cls, call):
        native = call.lowered
        return cls(call.family, call.variant, native.profile, native.implementation,
                   tuple(native.wire_port.physical_of_logical))


def _checked_call(call, algorithm, n, call_limits):
    native = call.lowered
    m = native.native_metrics
    if m.get('host') != algorithm.host.name or native.profile not in algorithm.host.profiles:
        raise PolicyRefusal("candidate belongs to another host or unavailable profile")
    if m.get('profile', native.profile) != native.profile:
        raise PolicyRefusal("native metrics identify another profile")
    if call.readout.width != n or call.readout.native_width != len(native.wire_port.physical_of_logical):
        raise PolicyRefusal("readout register or native port does not match")
    error = m.get('unitary_operator_error')
    if m.get('geometry_findings') is None or m['geometry_findings'] or not (
        isinstance(error, Real) and isfinite(error) and 0 <= error <= 1e-8
    ):
        raise PolicyRefusal("native schedule failed geometry or operator check")
    _check_costs(m, call_limits)


def _check_costs(metrics, limits):
    for key, maximum in limits.items():
        value = metrics.get(key)
        if not isinstance(value, Real) or not isfinite(value) or value < 0:
            raise PolicyRefusal(f"missing or invalid cost {key}")
        if value > maximum:
            raise PolicyRefusal(f"{key}={value} exceeds {maximum}")


def check_nonregression(candidate, reference, protected):
    """Require finite, nonnegative matched costs for every protected endpoint."""
    _check_costs(reference, {key: float("inf") for key in protected})
    _check_costs(candidate, {key: reference[key] for key in protected})


@dataclass(frozen=True)
class HostPolicy:
    host: str
    task_contract: str
    changed: bool
    actions: Mapping
    guards: Mapping
    call_limits: Mapping
    scope: str
    objective_order: tuple
    job_limits: Mapping
    success_floor: float
    retained_reference: bool = False

    def __post_init__(self):
        for name in ('actions', 'guards', 'call_limits', 'job_limits'):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def _guard(self, algorithm, task_contract, stage, ops, n):
        if self.host != algorithm.host.name or self.changed != algorithm.changed:
            raise PolicyRefusal("host or enabled compilation rules differ from extracted policy")
        if self.task_contract != task_contract or stage not in self.actions:
            raise PolicyRefusal("unrecognized task contract or stage")
        try:
            shape = _shape(ops, n)
        except (TypeError, ValueError) as exc:
            raise PolicyRefusal(f"invalid logical parameter binding: {exc}") from exc
        if shape not in self.guards[stage]:
            raise PolicyRefusal("logical register, topology or parameter slots differ from stage guard")
        action = self.actions[stage]
        if action.profile not in algorithm.host.profiles:
            raise PolicyRefusal("extracted host profile is unavailable")
        return action

    def choose(self, algorithm, task_contract, stage, ops, n, calls):
        """Select from a trusted pool already lowered for these exact ops and n."""
        action = self._guard(algorithm, task_contract, stage, ops, n)
        matching = [call for call in calls if Strategy.of(call) == action]
        if len(matching) != 1:
            raise PolicyRefusal(f"expected one lowered call for {action}, found {len(matching)}")
        _checked_call(matching[0], algorithm, n, self.call_limits)
        return matching[0]

    def bind(self, algorithm, task_contract, stage, ops, n, directory):
        """Rebind angles, lower on the named host and execute the extracted decision."""
        ops = tuple(ops)
        action = self._guard(algorithm, task_contract, stage, ops, n)
        mode = action.implementation.split('__port_', 1)[0]
        try:
            calls, rejected = algorithm.lower_instrument(
                ops, n, directory, profiles=(action.profile,), mode_filter=(mode,),
                allow_wire_relabel=action.wire_order != tuple(range(len(action.wire_order))))
        except (ValueError, RuntimeError, OSError) as exc:
            raise PolicyRefusal(f"host could not lower this binding: {exc}") from exc
        try:
            return self.choose(algorithm, task_contract, stage, ops, n, calls)
        except PolicyRefusal as exc:
            raise PolicyRefusal(f"{exc}; native lowering reported {rejected}") from exc

    def verify_job(self, result, reference=None):
        """Check new whole-job evidence using the extracted objective and limits.

        Paired verification compares against a supplied assessment of the matched
        reference. A retained reference is checked against the thresholds alone.
        A successful per-call bind alone only establishes its call guard.
        """
        if self.scope == 'paired' and not self.retained_reference and reference is None:
            raise PolicyRefusal("new paired verification requires a matched reference assessment")
        return _score(result, self.scope, self.objective_order, self.job_limits, self.success_floor,
                      self.task_contract, None if self.retained_reference else reference)

    def as_dict(self):
        return dict(version=1, host=self.host, task_contract=self.task_contract, changed=self.changed,
                    actions={k:asdict(a) for k,a in self.actions.items()}, guards=dict(self.guards),
                    call_limits=dict(self.call_limits), scope=self.scope, on_guard_miss='refuse',
                    objective_order=self.objective_order, job_limits=dict(self.job_limits),
                    success_floor=self.success_floor, retained_reference=self.retained_reference)

    @classmethod
    def from_dict(cls, data):
        if data.get('version') != 1 or data.get('on_guard_miss') != 'refuse':
            raise ValueError("Unsupported policy version or guard-miss instruction")
        actions = {k:Strategy(**dict(v, wire_order=tuple(v['wire_order']))) for k,v in data['actions'].items()}
        guards = {k:tuple((n, tuple((name, tuple(wires), angle) for name, wires, angle in word))
                          for n, word in shapes) for k,shapes in data['guards'].items()}
        if actions.keys() != guards.keys():
            raise ValueError("Each stage must have an action and a guard")
        return cls(data['host'], data['task_contract'], data['changed'], actions, guards,
                   data['call_limits'], data['scope'], tuple(data['objective_order']),
                   data['job_limits'], data['success_floor'], data.get('retained_reference', False))


@dataclass(frozen=True)
class Extraction:
    policy: HostPolicy | None
    status: str
    evaluated: int
    space_size: int
    rejected: tuple
    score: object = None

    @property
    def exhaustive(self):
        return self.evaluated == self.space_size


def _indexed(stages, algorithm, call_limits):
    options, guards, indexed, rejected = {}, {}, {}, []
    for stage, bindings in stages.items():
        if not bindings:
            raise ValueError(f"Stage {stage} has no bindings")
        table, shapes = {}, set()
        for key, binding in bindings.items():
            shapes.add(_shape(binding.ops, binding.n))
            candidates = {}
            for call in binding.calls:
                action = Strategy.of(call)
                try:
                    _checked_call(call, algorithm, binding.n, call_limits)
                except PolicyRefusal as exc:
                    rejected.append((stage, key, action, str(exc)))
                    continue
                if action in candidates:
                    raise ValueError(f"Ambiguous lowered action {action} for {stage} binding {key}")
                candidates[action] = call
            table[key] = candidates
        common = set.intersection(*(set(row) for row in table.values()))
        union = set.union(*(set(row) for row in table.values()))
        for action in sorted(union - common):
            missing = tuple(k for k,v in table.items() if action not in v)
            rejected.append((stage, missing, action, 'unavailable across stage bindings'))
        # Try lower worst-binding native duration first. It is only a search order.
        def priority(action):
            times = (row[action].lowered.native_metrics.get('duration_us') for row in table.values())
            return max(t if isinstance(t, Real) and isfinite(t) else float('inf') for t in times), action
        indexed[stage], options[stage], guards[stage] = table, sorted(common, key=priority), tuple(sorted(shapes))
    return indexed, options, guards, rejected


def _vectors(counts, reference):
    """Visit the ordinal lattice by sum of within-stage ranks without forming its product."""
    if reference is not None:
        yield reference
    zero = (0,) * len(counts)
    queue, seen = [(0, zero)], {zero}
    while queue:
        rank, v = heappop(queue)
        if v != reference:
            yield v
        for i, count in enumerate(counts):
            if v[i] + 1 < count:
                w = v[:i] + (v[i]+1,) + v[i+1:]
                if w not in seen:
                    seen.add(w)
                    heappush(queue, (rank+1, w))


def _score(result, scope, order, limits, success_floor, task_contract, base):
    if scope == 'paired':
        if not isinstance(result, Assessment) or result.task_contract != task_contract:
            raise ValueError("Paired evaluation requires an Assessment for the requested task")
        if base is not None:
            comparison = compare(result, base, success_floor)
            if not comparison['certified_pareto']:
                raise PolicyRefusal(f"no paired dominance under supplied evidence: {comparison}")
        success = result.success_probability
        if success is None or success.low < success_floor:
            raise PolicyRefusal("whole-job success floor lacks evidence")
        costs = {k:(v.high if v is not None else None) for k,v in result.objectives.items()}
        applicable = tuple(k for k in order if result.encoding == 'encoded' or k != 'logical_error')
    else:
        if not isinstance(result, Mapping):
            raise TypeError("Resource-model evaluation requires a mapping of whole-job costs")
        if success_floor:
            success = result.get('success_probability')
            if not isinstance(success, Real) or not isfinite(success) or not success_floor <= success <= 1:
                raise PolicyRefusal("whole-job success floor lacks evidence")
        costs, applicable = result, order
    _check_costs(costs, {k:float('inf') for k in applicable})
    _check_costs(costs, limits)
    return tuple(costs[k] for k in applicable)


def extract_policy(algorithm, task_contract, stages, evaluate, *, objective_order,
                   limits=None, call_limits=None, max_evaluations=256, scope='resource_model',
                   success_floor=0., reference=None, protected_costs=()):
    """Compare *complete* policies on a finite, supplied stage/binding envelope.

    stages maps each call site to {binding_id: Binding}. evaluate receives the
    same keys with one InstrumentCall per binding. In resource_model mode it
    returns nonnegative finite costs in objective_order (minimized lexically).
    In paired mode it returns an Assessment, and reference maps each stage to
    a Strategy. Improvement requires certified Pareto dominance of the reference.
    limits are whole-job upper bounds (use infidelity for a fidelity floor).
    call_limits are upper bounds on native metric fields, replayed on each bind.

    Every result is scoped to the supplied evaluation and binding coverage.
    A truncated ordinal-lattice traversal has no global approximation guarantee.
    """
    order = tuple(objective_order)
    if (not isinstance(task_contract, str) or not task_contract or not stages or
            any(not isinstance(s, str) or not s for s in stages) or not order or
            len(set(order)) != len(order)):
        raise ValueError("Expected a named task contract, named stages and unique cost priorities")
    if not isinstance(max_evaluations, Integral) or max_evaluations < 1:
        raise ValueError("Expected a positive whole-policy evaluation budget")
    if not 0 <= success_floor <= 1 or scope not in ('paired', 'resource_model'):
        raise ValueError("Invalid success floor or evaluation scope")
    if scope == 'paired' and (reference is None or set(order) not in (set(OBJECTIVES),set(PULSE_OBJECTIVES)|{'logical_error'})):
        raise ValueError("Paired evaluation requires a reference and all six cost priorities")
    if reference is not None and reference.keys() != stages.keys():
        raise ValueError("Reference must specify exactly one action per stage")
    protected_costs = tuple(protected_costs)
    if protected_costs and (reference is None or scope != "resource_model"):
        raise ValueError("Model non-regression requires a complete reference action")
    limits, call_limits = dict(limits or {}), dict(call_limits or {})
    for cap in (*limits.values(), *call_limits.values()):
        if not isinstance(cap, Real) or not isfinite(cap) or cap < 0:
            raise ValueError("Upper bounds must be finite and nonnegative")
    table, options, guards, rejected = _indexed(stages, algorithm, call_limits)
    names = tuple(stages)
    counts = tuple(len(options[s]) for s in names)
    space = prod(counts)
    if not space:
        missing = tuple(s for s in names if not options[s])
        return Extraction(None, f"no common admissible action at stages {missing}", 0, 0, tuple(rejected))
    ref = None
    if reference is not None:
        if any(reference[s] not in options[s] for s in names):
            return Extraction(None, "reference unavailable under stage guards", 0, space, tuple(rejected))
        ref = tuple(options[s].index(reference[s]) for s in names)
    evaluated = 0
    best_actions = best_score = best_result = base = None
    for vector in _vectors(counts, ref):
        actions = {s:options[s][i] for s,i in zip(names,vector)}
        selected = {s:{key:candidates[actions[s]] for key,candidates in table[s].items()} for s in names}
        evaluated += 1
        try:
            result = evaluate(selected)
            if protected_costs:
                _check_costs(result, {k:float('inf') for k in protected_costs})
                if vector == ref:
                    base = result
                else:
                    if base is None:
                        raise PolicyRefusal("Reference cost vector is unavailable")
                    check_nonregression(result, base, protected_costs)
            if scope == 'paired' and vector == ref:
                if not isinstance(result, Assessment) or result.task_contract != task_contract:
                    raise ValueError("Expected whole-job assessment of the reference for the requested task")
                base = result
            score = _score(result, scope, order, limits, success_floor, task_contract,
                           base if vector != ref else None)
        except PolicyRefusal as exc:
            rejected.append(('complete', vector, str(exc)))
            if scope == 'paired' and vector == ref and base is None:
                break
        else:
            if best_score is None or score < best_score:
                best_actions, best_score, best_result = actions, score, result
        if evaluated >= max_evaluations:
            break
    cover = 'exhaustive on supplied pool' if evaluated == space else 'partial search on supplied pool'
    if best_actions is None:
        return Extraction(None, f"no admissible whole-job policy; {cover}", evaluated, space, tuple(rejected))
    policy = HostPolicy(algorithm.host.name, task_contract, algorithm.changed, best_actions,
                        guards, call_limits, scope, order, limits, success_floor,
                        scope == 'paired' and best_actions == reference)
    if scope == 'resource_model':
        status = "provisional whole-job resource model"
    else:
        status = "retained reference" if best_actions == reference else "paired dominance under supplied evidence"
    return Extraction(policy, f"{status}; {cover}", evaluated, space, tuple(rejected), best_result)
