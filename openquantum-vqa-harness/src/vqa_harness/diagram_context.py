"""Type-check substitution of an open subdiagram in a finite acyclic host.

Ports are named linear wires. A fork, measurement, effect discard or classical
copy is an explicit operation. Types are the host's declared space and frame.
This layer checks composition. A task verifier must check the semantics and
device constraints before the enclosing complete policy can be released.
"""
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from itertools import chain


@dataclass(frozen=True)
class Port:
    name: str
    space: str

    def __post_init__(self):
        if not self.name or not self.space:
            raise ValueError("A port needs a wire name and a declared type")


@dataclass(frozen=True)
class Stage:
    id: str
    role: str
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    # Stable native instruction, calibration and classical-action descriptor.
    # Hosts using unique immutable implementation IDs may use the stage ID.
    implementation: str = ""

    def __post_init__(self):
        if not isinstance(self.implementation, str):
            raise TypeError("The physical implementation key must be a string")
        if not self.id or not self.role:
            raise ValueError("A stage needs an identity and an exposed role")


@dataclass(frozen=True)
class OpenDiagram:
    inputs: tuple[Port, ...]
    stages: tuple[Stage, ...]
    outputs: tuple[Port, ...]

    def __post_init__(self):
        # Every wire has exactly one source and one sink, including boundary ports.
        producers = {}
        declarations = {}
        sources = chain([(None, self.inputs)], ((s.id, s.outputs) for s in self.stages))
        for owner, endpoints in sources:
            for port in endpoints:
                if port.name in producers:
                    raise ValueError(f"Multiple sources for {port.name}")
                producers[port.name] = owner
                declarations[port.name] = port
        if len({s.id for s in self.stages}) != len(self.stages):
            raise ValueError("Duplicate stage identity")
        dependencies = {stage.id:set() for stage in self.stages}
        consumed = set()
        sinks = chain(((s.id, s.inputs) for s in self.stages), [(None, self.outputs)])
        for owner, endpoints in sinks:
            for port in endpoints:
                if declarations.get(port.name) != port or port.name in consumed:
                    raise ValueError(f"Absent, mistyped or multiply consumed wire {port.name}")
                consumed.add(port.name)
                source = producers[port.name]
                if owner is not None and source is not None:
                    dependencies[owner].add(source)
        if set(producers) != consumed:
            raise ValueError("Every linear wire needs a sink")
        try:
            order = tuple(TopologicalSorter(dependencies).static_order())
        except CycleError as exc:
            raise ValueError("A stage substitution created a cycle") from exc
        by_id = {stage.id:stage for stage in self.stages}
        object.__setattr__(self,"stages",tuple(by_id[i] for i in order))


def replace_region(host, selected_ids, replacement, exposed_roles):
    """Replace any selected set of stages, preserving every named crossing port.

    The replacement may have another number/order of internal stages. It can
    be reinserted only if the typed exterior agrees and the full graph remains
    acyclic and linear. No bound on the number of selected stages is imposed.
    """
    selected_ids = tuple(selected_ids)
    wanted = set(selected_ids)
    selected = tuple(s for s in host.stages if s.id in wanted)
    if not wanted or len(wanted) != len(selected_ids) or len(selected) != len(wanted):
        raise ValueError("Select distinct existing host stages")
    if any(s.role not in exposed_roles for s in (*selected,*replacement.stages)):
        raise ValueError("A stage lies outside the registered rewrite roles")
    consumed = {p for stage in selected for p in stage.inputs}
    produced = {p for stage in selected for p in stage.outputs}
    if set(replacement.inputs) != consumed-produced or set(replacement.outputs) != produced-consumed:
        raise ValueError("Replacement does not preserve the typed incoming and outgoing ports")
    if selected == replacement.stages:
        return host
    if len(selected) == len(replacement.stages) == 1:
        # Keep the host's scheduling priority among independent stages.
        revised = tuple(replacement.stages[0] if s.id in wanted else s for s in host.stages)
        return OpenDiagram(host.inputs, revised, host.outputs)
    outside = tuple(s for s in host.stages if s.id not in wanted)
    return OpenDiagram(host.inputs,(*outside,*replacement.stages),host.outputs)
