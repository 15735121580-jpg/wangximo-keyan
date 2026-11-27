"""A deterministic native CZ/rotation rewrite on preparation and readout ports."""
from math import cos, sin, atan2, pi, isfinite
from cmath import phase, exp
from copy import deepcopy
from itertools import groupby

I = (1, 0, 0, 1)
Z = (1, 0, 0, -1)
EPS = 1e-14


def multiply(a, b):
    return tuple(a[2*i]*b[j] + a[2*i+1]*b[2+j] for i in range(2) for j in range(2))


def local_matrix(g):
    p = tuple(g.get('params', ()))
    if not all(isfinite(x) for x in p):
        raise ValueError('Unbound rotation')
    if g['gate'] in ('u3', 'u'):
        t, a, b = p
    elif g['gate'] == 'u2':
        a, b = p
        t = pi/2
    elif g['gate'] in ('u1', 'p'):
        b, = p
        t, a = 0., 0.
    else:
        raise ValueError('Unsupported local primitive')
    c, s = cos(t/2), sin(t/2)
    return (c, -exp(1j*b)*s, exp(1j*a)*s, exp(1j*(a+b))*c)


def monomial(u):
    if max(abs(u[1]), abs(u[2])) <= EPS:
        return 0
    if max(abs(u[0]), abs(u[3])) <= EPS:
        return 1
    return None


def project_frame(u, flip):
    # The residual from projecting near-degenerate floating-point angles is
    # bounded separately by independent full-state checks after the freeze.
    if flip:
        return (0, u[1]/abs(u[1]), u[2]/abs(u[2]), 0)
    return (u[0]/abs(u[0]), 0, 0, u[3]/abs(u[3]))


def euler(u):
    # Called only on a non-monomial, modulo a scalar phase.
    return [2*atan2(abs(u[2]), abs(u[0])), phase(u[2]/u[0]), phase(-u[1]/u[0])]


def metrics(gates, n):
    layers = [0]*n
    for op in gates:
        wires = op['qubits']
        end = 1 + max(layers[q] for q in wires)
        for q in wires:
            layers[q] = end
    cz = sum(op['gate']=='cz' for op in gates)
    return dict(native_gates=len(gates), local_gates=len(gates)-cz, cz=cz,
                gate_dependency_depth=max(layers, default=0), native_address_width=n)


def rewrite_native_ports(source, *, allow_local_frames=False, allow_skipping_cz_runs=False):
    """Input is a bound, reset-zero, terminal-Z-sampled native program.

    allow_local_frames affirms arbitrary per-wire U3 retuning, independently
    disableable local actions and post-readout XOR support. allow_skipping_cz_runs
    affirms that entangling pulses can be disabled while retaining all transport
    and that each entangling action is wholly contained in one contiguous CZ run.
    All indices in source_slots refer to the untouched chronological source.
    Charges are per invocation of the supplied native circuit.
    """
    n = source['num_qubits']
    gates = source['gates']
    for op in gates:
        wires = op['qubits']
        if len(set(wires))!=len(wires) or any(not isinstance(q, int) or q not in range(n) for q in wires):
            raise ValueError('Invalid native gate ports')
        if op['gate']=='cz':
            if len(wires)!=2 or op.get('params',[]):
                raise ValueError('Invalid CZ')
        else:
            if len(wires)!=1:
                raise ValueError('Unsupported native gate')
            local_matrix(op)
    basis = [0]*n
    frames = [I]*n
    actions, kept, idle, virtual = [], [], [], []
    for cz_run, block in groupby(enumerate(gates), key=lambda item: item[1]['gate']=='cz'):
        group = list(block)
        if cz_run and allow_skipping_cz_runs and all(any(basis[q]==0 for q in g['qubits']) for _,g in group):
            idle.extend(i for i,_ in group)
            continue
        for i, op in group:
            if cz_run:
                a, b = op['qubits']
                fa, fb = monomial(frames[a]), monomial(frames[b])
                if fa:
                    frames[b] = multiply(Z, frames[b])
                if fb:
                    frames[a] = multiply(Z, frames[a])
            else:
                q, = op['qubits']
                u = local_matrix(op)
                flip = monomial(u)
                if basis[q] is not None:
                    basis[q] = None if flip is None else basis[q]^flip
                if allow_local_frames:
                    if flip is not None:
                        frames[q] = multiply(project_frame(u, flip), frames[q])
                        virtual.append(i)
                        continue
                    if frames[q] != I:
                        rewritten = deepcopy(op)
                        rewritten.update(gate='u3', params=euler(multiply(u, frames[q])))
                        op = rewritten
                        frames[q] = I
            kept.append(i)
            actions.append(deepcopy(op))
    program = {**deepcopy(source), 'gates':actions}
    # A CZ conjugation can contribute a scalar phase. The contract is measured.
    if 'global_phase' in program:
        program.pop('global_phase')
    return dict(native_program=program, bit_xor_mask=sum(monomial(f)<<q for q,f in enumerate(frames)),
                source_slots=kept, omitted_slots={'zero_controlled_cz':idle, 'monomial_local':virtual},
                metrics=metrics(actions,n),
                preconditions={'zero_preparation':True, 'terminal_computational_readout':True,
                               'allow_local_frames':allow_local_frames,
                               'allow_skipping_cz_runs':allow_skipping_cz_runs})
