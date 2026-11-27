"""Construct policies from registered relations on an entire open host diagram.

The host adapter supplies pure relation matchers and a trusted verifier. Candidate
identity includes physical operation keys, named wires, frame/score interfaces
and the controller. Matches must depend only on the current diagram and the
frozen task/device contract. All candidate scores concern complete programs.
"""
from collections import deque
from dataclasses import dataclass

from .diagram_context import OpenDiagram, replace_region
from .guarded_release import guarded_release
from .host_compilation import direct_receipt, incumbent_policy


@dataclass(frozen=True)
class Match:
    selected: tuple[str, ...]
    replacement: OpenDiagram


def _check(job, relations, limit):
    if not isinstance(job.host, OpenDiagram):
        raise TypeError("Diagram search takes an imported OpenDiagram")
    if type(limit) is not int or limit < 0:
        raise ValueError("Search limits must be nonnegative integers")
    names = [name for name, _ in relations]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Give each registered relation a distinct name")


def _rewrites(job, graph, relations):
    for name, match in relations:
        for occurrence in match(job, graph):
            rewritten = replace_region(graph, occurrence.selected,
                                       occurrence.replacement, job.stages)
            if rewritten != graph:
                yield (name, occurrence.selected), rewritten


def direct_diagram(job, relations, verifier, max_rewrites):
    """Construct one complete policy using the registered priority and check it once."""
    relations = tuple(relations)
    _check(job, relations, max_rewrites)
    steps = []

    def construct():
        graph = job.host
        seen = {graph}
        for _ in range(max_rewrites):
            choice = next(((step, nxt) for step, nxt in _rewrites(job, graph, relations)
                           if nxt not in seen), None)
            if choice is None:
                break
            step, graph = choice
            seen.add(graph)
            steps.append(step)
        if steps:
            yield graph

    result = guarded_release(job.contract, job.host, construct(), verifier)
    return dict(result, proposed_steps=steps, incumbent=direct_receipt(job, result, verifier, steps if result["status"]=="released" else ()))


def search_diagram(job, relations, verifier, max_rewrites, max_candidates, *, incumbent=None, legacy_host_only=False):
    """Breadth-first closure to the registered rewrite depth or candidate budget.

    A partial rewrite with poor physical costs may be followed by a useful
    rewrite. Only duplicate diagrams suppress expansion. Budgets count distinct
    non-host diagrams. Pass max_rewrites=None to explore until the registered
    relations reach a fixed point or the candidate budget interrupts the search.
    In that mode registered_closure_complete reports actual fixed-point coverage.
    By default the direct priority policy, or a supplied
    direct receipt, remains available even outside that budget. Exact
    certification additionally requires exact scores,
    sound semantic checks and the advertised coverage of the physical library.
    """
    relations = tuple(relations)
    if type(max_candidates) is not int or max_candidates < 0:
        raise ValueError("The candidate budget must be a nonnegative integer")
    direct_limit = max_candidates if max_rewrites is None else max_rewrites
    _check(job, relations, direct_limit)
    parents = {job.host: None}
    candidates = []
    exhausted = False

    def construct():
        nonlocal exhausted
        exhausted = True
        queue = deque([(job.host, 0)])
        while queue:
            graph, depth = queue.popleft()
            if depth == max_rewrites:
                continue
            for step, nxt in _rewrites(job, graph, relations):
                if nxt in parents:
                    continue
                if len(candidates) == max_candidates:
                    exhausted = False
                    queue.clear()
                    break
                parents[nxt] = (graph, step)
                candidates.append(nxt)
                queue.append((nxt, depth + 1))
        # Freeze the proposed set before inspecting any candidate scores.
        yield from candidates

    if type(legacy_host_only) is not bool or (legacy_host_only and incumbent is not None):
        raise ValueError("Choose the protected incumbent or explicit host-only replay")
    ticket = None
    if legacy_host_only:
        result = guarded_release(job.contract, job.host, construct(), verifier)
    else:
        if incumbent is None:
            # The registered priority derivation defines the canonical direct policy.
            incumbent = direct_diagram(job, relations, verifier, direct_limit)
        ticket = incumbent_policy(job, incumbent, verifier)
        result = guarded_release(job.contract, job.host, construct(), verifier,
                                 incumbent=None if ticket is None else ticket.policy)
    program = result["policy"]
    if result["status"] == "retained_incumbent":
        steps = list(ticket.steps)
    else:
        steps = []
        while program is not None and parents[program] is not None:
            program, step = parents[program]
            steps.append(step)
        steps.reverse()
    result.update(selected_steps=steps, evaluated=len(result["branches"]),
                  constructed=len(candidates), exhausted_to_depth=exhausted, depth=max_rewrites)
    if max_rewrites is None:
        result["registered_closure_complete"] = exhausted
    return result
