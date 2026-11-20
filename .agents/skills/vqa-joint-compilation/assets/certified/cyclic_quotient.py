"""Strong probabilistic bisimulation for the finite cyclic VQA job backend.

Actions share *public names*, while their physical implementations are local
to each source port. Preserve duration, peak, depth, each (output,fault) mass,
and total successor mass into every block. With optional public colors these
tests refine the indiscrete color partition to the coarsest stable partition.
They imply the commuting square for one action; composition preserves it.

With the independent private randomization permitted by cyclic_compiler, any
proper history-based source controller has a quotient controller with the same
performance. Sum its finite expected action occupations over each block (and
spent-depth value): the exact square preserves flow balance and terminal
statistics. Every positive summed occupation comes from a finite path of
positive probability. Its projection is reachable using the actions retained
by Markovization. Hence the summed occupations contain no unreachable
circulation, and their normalization gives a quotient policy with equal costs.
Equivalently, the quotient controller can sample a private fine-state history
conditional on its observable block history, then imitate the fine policy.
Conversely each quotient policy lifts by reading the block, and invoking the
fine state's own physical realization for the chosen action name. This argument
requires the finite state/goal/observation/cost sufficiency of the given Job.
It does not grant a deterministic controller a previously unavailable coin.
The cost interface records the kernel's conditional mean. To expose elapsed
time to the controller the backend must also include it in the public state.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction

import z3

from cyclic_compiler import (Action, Job, Outcome, Solution, State, Statistics,
                             _validate, compile_stopping_job, replay, resource_graph)


@dataclass(frozen=True)
class Bisimulation:
    original: Job
    quotient: Job
    projection: dict[str, str]
    members: dict[str, tuple[str, ...]]
    local_actions: dict[tuple[str, str], Action]
    colors: dict[str, Hashable]


@dataclass(frozen=True)
class Compiled:
    certificate: Bisimulation
    quotient_solution: Solution
    policy: dict[State, dict[str, z3.ArithRef]] | None
    original_statistics: Statistics | None


def _colors(j: Job, colors: Mapping[str, Hashable] | None):
    if colors is None:
        return {s: None for s in j.actions}
    if set(colors) != set(j.actions):
        raise ValueError("public port colors must cover exactly the original states")
    for c in colors.values():
        hash(c)
    return dict(colors)


def _masses(action: Action, project) -> tuple:
    """Record terminal output and fault flag jointly, never as separate marginals."""
    masses = defaultdict(Fraction)
    for o in action.outcomes:
        key = ((0, project[o.next_state], False) if o.next_state is not None
               else (1, o.output, o.failed))
        masses[key] += Fraction(o.probability)
    return tuple(sorted(masses.items()))


def _actsig(action: Action, project):
    return (action.name, Fraction(action.duration), action.peak_atoms, action.depth,
            _masses(action, project))


def _signature(actions, project):
    return tuple(sorted(_actsig(a, project) for a in actions))


def _labels(entries, states):
    labels = {}
    lookup = {}
    for s in states:
        sig = entries[s]
        lookup.setdefault(sig, len(lookup))
        labels[s] = lookup[sig]
    return labels


def _partition(j: Job, colors):
    states = tuple(sorted(j.actions))
    part = _labels(colors, states)
    while True:
        sig = {s: (part[s], _signature(j.actions[s], part)) for s in states}
        nxt = _labels(sig, states)
        if nxt == part:
            return {s: f"block_{part[s]}" for s in states}
        part = nxt


def bisimulation(j: Job, colors: Mapping[str, Hashable] | None = None) -> Bisimulation:
    _validate(j)
    color = _colors(j, colors)
    projection = _partition(j, color)
    grouped = defaultdict(list)
    for s, block in projection.items():
        grouped[block].append(s)
    members = {b: tuple(ss) for b, ss in grouped.items()}
    ops = {}
    for block, ss in members.items():
        coarse_actions = []
        for a in j.actions[ss[0]]:
            outcomes = []
            for (kind, label, failed), mass in _masses(a, projection):
                outcomes.append(Outcome(mass, next_state=label) if kind == 0
                                else Outcome(mass, output=label, failed=failed))
            coarse_actions.append(replace(a, outcomes=tuple(outcomes)))
        ops[block] = tuple(coarse_actions)
    coarse = replace(j, start=projection[j.start], actions=ops, ideal=dict(j.ideal))
    cert = Bisimulation(j, coarse, projection, members,
                        {(s, a.name): a for s, ac in j.actions.items() for a in ac},
                        color)
    if not verify_square(cert):
        raise ArithmeticError("constructed partition failed its commuting-square check")
    return cert


def verify_square(c: Bisimulation) -> bool:
    """Check *every* original action, full support, observation, resource and lift."""
    j, q, p = c.original, c.quotient, c.projection
    try:
        _validate(j)
        _validate(q)
        if (set(p) != set(j.actions) or set(p.values()) != set(q.actions)
                or set(c.members) != set(q.actions) or set(c.colors) != set(p)
                or q.start != p[j.start] or j.ideal != q.ideal
                or (j.max_atoms, j.max_depth, j.error_limit, j.fidelity_min) !=
                   (q.max_atoms, q.max_depth, q.error_limit, q.fidelity_min)):
            return False
        flat = [s for ss in c.members.values() for s in ss]
        if len(flat) != len(p) or set(flat) != set(p):
            return False
        if set(c.local_actions) != {(s, a.name) for s, ops in j.actions.items() for a in ops}:
            return False
        identity = {b: b for b in q.actions}
        for block, ss in c.members.items():
            if (not ss or any(p[s] != block for s in ss)
                    or any(c.colors[s] != c.colors[ss[0]] for s in ss)):
                return False
            coarse = _signature(q.actions[block], identity)
            for s in ss:
                if _signature(j.actions[s], p) != coarse:
                    return False
                if any(c.local_actions[s, a.name] is not a for a in j.actions[s]):
                    return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def lift_policy(c: Bisimulation, policy: dict[State, dict[str, z3.ArithRef]]):
    if not verify_square(c):
        raise ValueError("uncertified quotient")
    out = {}
    for concrete in resource_graph(c.original):
        s, k = concrete
        law = policy.get((c.projection[s], k))
        if law is not None:
            # Same public name, but the emitted action is implemented at s.
            out[concrete] = {c.local_actions[s, name].name: prob for name, prob in law.items()}
    return out


def _equal(x, y):
    return z3.is_true(z3.simplify(x == y))


def compile_quotient(j: Job, colors: Mapping[str, Hashable] | None = None) -> Compiled:
    c = bisimulation(j, colors)
    answer = compile_stopping_job(c.quotient)
    if answer.policy is None:
        return Compiled(c, answer, None, None)
    policy = lift_policy(c, answer.policy)
    stats = replay(j, policy)  # original branch law and original resource monitor
    old = answer.statistics
    if (not _equal(stats.expected_time, old.expected_time) or not _equal(stats.failure, old.failure)
            or not _equal(stats.fidelity, old.fidelity)
            or any(not _equal(stats.output[y], old.output[y]) for y in j.ideal)):
        raise ArithmeticError("original policy replay does not reproduce quotient metrics")
    masses = {}
    for (state, depth), value in stats.visits.items():
        key = c.projection[state], depth
        masses[key] = masses.get(key, 0) + value
    if set(masses) != set(old.visits) or any(
            not _equal(masses[key], count) for key, count in old.visits.items()):
        raise ArithmeticError("summed fine visits disagree with quotient visit counts")
    return Compiled(c, answer, {s: policy[s] for s in stats.visits}, stats)
