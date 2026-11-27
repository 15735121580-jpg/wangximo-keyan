"""Compose host-supplied implementations along the typed wires of a program."""

from .diagram_context import OpenDiagram, Port, Stage
from .diagram_search import Match


def _implementation_options(job, graph, stage, alternatives):
    options = {}
    for candidate in alternatives(job, graph, stage):
        if not isinstance(candidate, Stage):
            raise TypeError(f"Expected a Stage implementation for {stage.id}")
        if candidate.id != stage.id or candidate.role != stage.role:
            raise ValueError(f"Implementation changes the identity or role of {stage.id}")
        for old, new in ((stage.inputs, candidate.inputs), (stage.outputs, candidate.outputs)):
            if (not isinstance(new, tuple) or len(old) != len(new)
                    or any(not isinstance(p, Port) or p.name != ref.name
                           or not isinstance(p.space, str) for ref, p in zip(old, new))):
                raise ValueError(f"Implementation changes the named interfaces of {stage.id}")
        options[candidate] = None
    options[stage] = None
    return tuple(options)


def _region(changed):
    inputs = tuple(p for s in changed for p in s.inputs)
    outputs = tuple(p for s in changed for p in s.outputs)
    consumed, produced = set(inputs), set(outputs)
    replacement = OpenDiagram(tuple(p for p in inputs if p not in produced), changed,
                              tuple(p for p in outputs if p not in consumed))
    return Match(tuple(s.id for s in changed), replacement)


def implementation_relation(alternatives):
    """Build a matcher for a host's finite, ordered local implementation catalogs.

    `alternatives(job, graph, stage)` supplies Stage choices for each exposed role.
    A choice preserves the stage ID, role and ordered wire names, but may change
    the physical implementation and the types on internal wires. The current
    implementation is always an option. Supplied choices have priority; ties in
    a complete assignment follow the graph's topological stage order. No score
    is queried here. Use the resulting matcher with direct_diagram/search_diagram
    and the host's verifier of semantics, feasibility and complete-job costs.

    Every compatible non-identity assignment yields one Match. All changed
    stages are substituted together. A register, frame or classical record can
    therefore change on an internal edge even when neither endpoint can change
    independently. Candidate counts can grow exponentially for independent
    choices; assignments are streamed, and fixed stages do not add search levels.
    """
    if not callable(alternatives):
        raise TypeError("Provide a callable for the stage implementations")

    def match(job, graph):
        fixed = {p.name: p.space for p in (*graph.inputs, *graph.outputs)}
        stages, catalogs = [], []
        for stage in graph.stages:
            options = (_implementation_options(job, graph, stage, alternatives)
                       if stage.role in job.stages else (stage,))
            if len(options) == 1:
                fixed.update((p.name, p.space) for p in (*stage.inputs, *stage.outputs))
            else:
                stages.append(stage)
                catalogs.append(options)
        if not stages:
            return

        # Enforce all fixed neighbours and job boundaries before backtracking.
        for i, options in enumerate(catalogs):
            catalogs[i] = tuple(s for s in options if all(
                fixed.get(p.name, p.space) == p.space for p in (*s.inputs, *s.outputs)))

        # Each variable input has either a fixed boundary or a unique preceding
        # producer. Backtracking can leave old output entries here: the producer
        # always overwrites its entries before a downstream choice reads them.
        wire_types = fixed.copy()
        chosen = list(stages)
        cursors = [0] * len(stages)
        i = 0
        while i >= 0:
            if cursors[i] == len(catalogs[i]):
                cursors[i] = 0
                i -= 1
                continue
            option = catalogs[i][cursors[i]]
            cursors[i] += 1
            if any(wire_types[p.name] != p.space for p in option.inputs):
                continue
            chosen[i] = option
            wire_types.update((p.name, p.space) for p in option.outputs)
            if i + 1 < len(stages):
                i += 1
                continue
            changed = tuple(s for s, current in zip(chosen, stages) if s != current)
            if changed:
                yield _region(changed)

    return match
