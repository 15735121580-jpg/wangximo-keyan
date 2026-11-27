"""Backward Pauli-test compilation. Run with Python 3.9 or newer."""
import argparse
import itertools
import json
import math
from collections import deque
from pathlib import Path


# For U, each pair specifies U†X_i U and U†Z_i U in q0-first order.
GENERATORS = {
    'h': [('Z', 'X')], 's': [('-Y', 'Z')], 'sdg': [('Y', 'Z')],
    'x': [('X', '-Z')], 'y': [('-X', '-Z')], 'z': [('-X', 'Z')],
    'cx': [('XX', 'ZI'), ('IX', 'ZZ')],
    'cz': [('XZ', 'ZI'), ('ZX', 'IZ')],
    'swap': [('IX', 'IZ'), ('XI', 'ZI')],
}
ROTATIONS = {'rx': 'X', 'ry': 'Y', 'rz': 'Z', 'pauli_rotation': None}
CYCLIC = {'XY': 'Z', 'YZ': 'X', 'ZX': 'Y'}
SUPPORT_LIMIT = 4096


def multiply(left, right):
    """Multiply unsigned Pauli words. The signed result is i**phase times word."""
    word = []
    phase = 0
    for a, b in zip(left, right):
        if a == 'I':
            word.append(b)
        elif b == 'I':
            word.append(a)
        elif a == b:
            word.append('I')
        elif a + b in CYCLIC:
            word.append(CYCLIC[a + b])
            phase += 1
        else:
            word.append(CYCLIC[b + a])
            phase -= 1
    return phase % 4, ''.join(word)


def clifford_table(generators):
    lookup = {}
    for local in itertools.product('IXYZ', repeat=len(generators)):
        phase, word = 0, 'I' * len(local)
        for pauli, (x, z) in zip(local, generators):
            if pauli == 'Y':
                phase += 1  # Y = i X Z
            factors = {'I': (), 'X': (x,), 'Z': (z,), 'Y': (x, z)}[pauli]
            for factor in factors:
                if factor.startswith('-'):
                    phase += 2
                    factor = factor[1:]
                extra, word = multiply(word, factor)
                phase += extra
        assert phase % 2 == 0
        lookup[''.join(local)] = (phase % 4 == 0, word)
    return lookup


TABLES = {op: clifford_table(gens) for op, gens in GENERATORS.items()}


def pauli_word(word, size):
    if not isinstance(word, str) or len(word) != size or set(word) - set('IXYZ'):
        raise ValueError('Invalid Pauli word or width')
    return word


def integer(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validate(task, device):
    n = task['n_qubits']
    count = device['n_qubits']
    if not integer(n) or not integer(count) or not 1 <= n <= count:
        raise ValueError('Both input and reference must fit the supplied device')
    bits = task['initial_bits']
    if len(bits) != n or any(not integer(b) or b not in (0, 1) for b in bits):
        raise ValueError('Expected one initial computational-basis bit per qubit')
    if not task['observables']:
        raise ValueError('At least one terminal observable is required')
    for obs in task['observables']:
        if not isinstance(obs['label'], str):
            raise ValueError('Expected a string observable label')
        for term in obs['terms']:
            pauli_word(term['pauli'], n)
            c = term['coeff']
            if type(c) not in (float, int) or not math.isfinite(c):
                raise ValueError('Observable must have finite real coefficients')
    for gate in task['gates']:
        op, qs = gate['op'], gate['qubits']
        if not qs or any(not integer(q) or not 0 <= q < n for q in qs) or len(set(qs)) != len(qs):
            raise ValueError('Invalid gate operands')
        if op in TABLES:
            if len(qs) != len(GENERATORS[op]):
                raise ValueError('Invalid Clifford arity')
        elif op in ROTATIONS:
            if op == 'pauli_rotation':
                pauli_word(gate['pauli'], len(qs))
            elif len(qs) != 1:
                raise ValueError('Expected one rotation operand')
            angle = gate['angle']
            if not ((isinstance(angle, str) and angle.strip()) or
                    (type(angle) in (int, float) and math.isfinite(angle))):
                raise ValueError('Invalid rotation angle')
        else:
            raise ValueError('Unsupported operation: ' + str(op))
    native = set(device['native'])
    if not {'rz', 'sx', 'x'} <= native or not native & {'cx', 'cz'}:
        raise ValueError('Native basis requires rz, sx, x and cx or cz')
    graph = [set() for _ in range(count)]
    for a, b in device['edges']:
        if not all(integer(v) and 0 <= v < count for v in (a, b)) or a == b:
            raise ValueError('Invalid undirected device edge')
        graph[a].add(b)
        graph[b].add(a)
    if len(breadth_first(graph, 0)) != count:
        raise ValueError('This interface requires a connected device')
    return graph


def breadth_first(graph, start):
    parents = {start: None}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        for w in sorted(graph[v]):
            if w not in parents:
                parents[w] = v
                queue.append(w)
    return parents


def scatter(word, qs, local):
    new = list(word)
    for q, p in zip(qs, local):
        new[q] = p
    return ''.join(new)


def step(words, gate):
    """Return a sound support over-approximation and gate-erasure status."""
    op, qs = gate['op'], gate['qubits']
    out = set()
    unchanged = True
    for word in words:
        local = ''.join(word[q] for q in qs)
        if op in TABLES:
            positive, new = TABLES[op][local]
            out.add(scatter(word, qs, new))
            unchanged &= positive and new == local
        else:
            generator = ROTATIONS[op] or gate['pauli']
            anticommutes = sum(a != b and 'I' not in (a, b)
                               for a, b in zip(local, generator)) % 2
            out.add(word)
            if anticommutes:
                _, product = multiply(local, generator)
                out.add(scatter(word, qs, product))
                unchanged = False
    return out, unchanged


def rewrite(task):
    words = {t['pauli'] for o in task['observables'] for t in o['terms'] if t['coeff'] != 0}
    conservative = None
    cert = []
    kept = []
    for i in reversed(range(len(task['gates']))):
        gate = task['gates'][i]
        if conservative is None and len(words) > SUPPORT_LIMIT:
            conservative = {q for w in words for q, p in enumerate(w) if p != 'I'}
        if conservative is None:
            before = len(words)
            words, skip = step(words, gate)
            evidence = {'suffix_paulis': before, 'prefix_paulis': len(words)}
        else:
            touched = set(gate['qubits'])
            skip = conservative.isdisjoint(touched)
            if not skip:
                conservative.update(touched)
            evidence = {'conservative_cone_size': len(conservative)}
        if not skip:
            kept.append(i)
        cert.append({'index': i, 'action': 'erase' if skip else 'retain', **evidence})
    kept.reverse()
    active = {q for i in kept for q in task['gates'][i]['qubits']}
    active.update(q for o in task['observables'] for t in o['terms'] if t['coeff'] != 0
                  for q, p in enumerate(t['pauli']) if p != 'I')
    wires = sorted(active)
    positions = {q: j for j, q in enumerate(wires)}
    gates = [dict(task['gates'][i], qubits=[positions[q] for q in task['gates'][i]['qubits']])
             for i in kept]
    obs = [{'label': o['label'], 'terms': [
        {'pauli': ''.join(t['pauli'][q] for q in wires), 'coeff': t['coeff']}
        for t in o['terms'] if t['coeff'] != 0]} for o in task['observables']]
    return {'n_qubits': len(wires), 'initial_bits': [task['initial_bits'][q] for q in wires],
            'gates': gates, 'observables': obs}, wires, list(reversed(cert))


def native_program(task, device, graph):
    """One common deterministic lowering for both arms. Allow global phase."""
    native = set(device['native'])
    layout = list(breadth_first(graph, 0))[:task['n_qubits']]
    code = []

    def emit(op, qs, angle=None):
        gate = {'op': op, 'qubits': list(qs)}
        if angle is not None:
            gate['angle'] = angle
        code.append(gate)

    def h(q):
        emit('rz', [q], math.pi / 2)
        emit('sx', [q])
        emit('rz', [q], math.pi / 2)

    def local_cx(a, b):
        if 'cx' in native:
            emit('cx', [a, b])
        else:
            h(b)
            emit('cz', [a, b])
            h(b)

    def swap(a, b):
        local_cx(a, b)
        local_cx(b, a)
        local_cx(a, b)

    def route(a, b):
        parents = breadth_first(graph, a)
        path = [b]
        while path[-1] != a:
            path.append(parents[path[-1]])
        path.reverse()
        return path

    def controlled(a, b, op):
        path = route(a, b)
        shifts = list(zip(path[:-2], path[1:-1]))
        for u, v in shifts:
            swap(u, v)
        u, v = path[-2:]
        if op == 'cx':
            local_cx(u, v)
        elif 'cz' in native:
            emit('cz', [u, v])
        else:
            h(v)
            local_cx(u, v)
            h(v)
        for u, v in reversed(shifts):
            swap(u, v)

    def basis_change(p, q, inverse):
        if p == 'X':
            h(q)
        if p == 'Y':
            if inverse:
                h(q)
                emit('rz', [q], math.pi / 2)
            else:
                emit('rz', [q], -math.pi / 2)
                h(q)

    def rotate(word, qs, angle):
        support = [(p, q) for p, q in zip(word, qs) if p != 'I']
        if not support:  # An identity rotation is a global phase.
            return
        for p, q in support:
            basis_change(p, q, False)
        target = support[-1][1]
        for _, q in support[:-1]:
            controlled(q, target, 'cx')
        emit('rz', [target], angle)
        for _, q in reversed(support[:-1]):
            controlled(q, target, 'cx')
        for p, q in reversed(support):
            basis_change(p, q, True)

    for q, bit in zip(layout, task['initial_bits']):
        if bit:
            emit('x', [q])
    for gate in task['gates']:
        op = gate['op']
        qs = [layout[q] for q in gate['qubits']]
        a = qs[0]
        if op in ('cx', 'cz'):
            controlled(*qs, op)
        elif op == 'swap':
            a, b = qs
            controlled(a, b, 'cx')
            controlled(b, a, 'cx')
            controlled(a, b, 'cx')
        elif op in ROTATIONS:
            rotate(ROTATIONS[op] or gate['pauli'], qs, gate['angle'])
        elif op == 'h':
            h(a)
        elif op == 'x':
            emit('x', [a])
        elif op == 'y':
            emit('x', [a])
            emit('rz', [a], math.pi)
        else:
            emit('rz', [a], {'z': math.pi, 's': math.pi/2, 'sdg': -math.pi/2}[op])

    depths = [0] * len(graph)
    active = set(layout)  # Includes any measured or initially occupied logical wires.
    two_qubit = 0
    for g in code:
        qs = g['qubits']
        active.update(qs)
        assert g['op'] in native
        if len(qs) == 2:
            assert qs[1] in graph[qs[0]]
            two_qubit += 1
        depth = 1 + max(depths[q] for q in qs)
        for q in qs:
            depths[q] = depth
    return {'logical_to_physical': layout, 'instructions': code,
            'instruction_count': len(code), 'two_qubit_count': two_qubit,
            'depth': max(depths, default=0), 'allocated_sites': len(active),
            'physical_observables': [
                {'label': o['label'], 'terms': [
                    {'coeff': t['coeff'], 'pauli': scatter('I'*len(graph),layout,t['pauli'])}
                    for t in o['terms']]} for o in task['observables']]}


def compile_task(task, device):
    graph = validate(task, device)
    smaller, wires, proof = rewrite(task)
    reference = native_program(task, device, graph)
    candidate = native_program(smaller, device, graph)
    metrics = ('two_qubit_count', 'instruction_count', 'depth', 'allocated_sites')
    trial_costs = {k: candidate[k] for k in metrics}
    admit = all(candidate[k] <= reference[k] for k in metrics)
    if not admit:
        smaller = task
        wires = list(range(task['n_qubits']))
    return {'format': 'pauli-test-compiler-v1', 'optimized_task': smaller,
            'source_wire_for_optimized_index': wires, 'certificate': proof,
            'rewrite_admitted': admit, 'trial_native_costs': trial_costs,
            'reference_native': reference, 'optimized_native': candidate if admit else reference}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('task', type=Path)
    cli.add_argument('device', type=Path)
    cli.add_argument('--out', type=Path, required=True)
    args = cli.parse_args()
    result = compile_task(json.loads(args.task.read_text()), json.loads(args.device.read_text()))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
