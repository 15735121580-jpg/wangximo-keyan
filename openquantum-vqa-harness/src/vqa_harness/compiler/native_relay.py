"""Rewrite a typed two-wire region with one RZZ and local unitaries to U/CZ.

No task labels, output distributions or candidate evaluations enter this pass.
The deterministic target is the second wire of the frontend's RZZ instruction.
"""
from math import atan2
import numpy as np
H = np.array([[1,1],[1,-1]], complex)/np.sqrt(2)
LOCAL = frozenset(('h','rx','ry','rz'))

def rotation(name, angle):
    if name == 'h':
        if angle is not None:
            raise ValueError('H has no parameter')
        return H
    if angle is None or not np.isfinite(angle):
        raise ValueError('Rotations require a finite angle')
    c, s = np.cos(angle/2), np.sin(angle/2)
    if name == 'rx':
        return np.array([[c,-1j*s],[-1j*s,c]])
    if name == 'ry':
        return np.array([[c,-s],[s,c]])
    return np.diag([np.exp(-0.5j*angle),np.exp(0.5j*angle)])

def angles_u(m):
    # U(theta,phi,lambda), with the global phase carried by m[0,0].
    a,b,c,d=m.flat
    theta=2*atan2(abs(c),abs(a))
    if abs(c)<1e-14:
        return [0.,0.,float(np.angle(d/a))]
    if abs(a)<1e-14:
        return [float(np.pi),0.,float(np.angle(-b/c))]
    return [theta,float(np.angle(c/a)),float(np.angle(-b/a))]

def compile_native_relay(source):
    pending = [np.eye(2, dtype=complex) for _ in range(2)]
    word = []
    def flush():
        for q, mat in enumerate(pending):
            if np.allclose(mat, mat[0, 0]*np.eye(2), atol=1e-12, rtol=0):
                pending[q] = np.eye(2, dtype=complex)
                continue
            angles = angles_u(mat)
            word.append(dict(gate='u', qubits=[q], params=list(map(float, angles))))
            pending[q] = np.eye(2, dtype=complex)

    rzz_count = 0
    measured = []
    for gate, qs, angle in source:
        if gate == 'measure':
            if len(qs)!=1 or qs[0] not in (0,1) or angle is not None:
                raise ValueError('Invalid terminal readout')
            measured.extend(qs)
            continue
        if measured:
            raise ValueError('A terminal destructive measurement must end the region')
        if gate in LOCAL and len(qs)==1 and qs[0] in (0,1):
            u = rotation(gate, angle)
            pending[qs[0]] = u @ pending[qs[0]]
        elif gate == 'rzz' and set(qs)=={0,1} and len(qs)==2:
            rzz_count += 1
            target = qs[1]
            pending[target] = H @ pending[target]
            flush()
            word.append(dict(gate='cz', qubits=list(qs), params=[]))
            pending[target] = rotation('rx',angle)
            flush()
            word.append(dict(gate='cz', qubits=list(qs), params=[]))
            pending[target] = H
        else:
            raise ValueError(f'Unsupported typed region {gate} on {qs}')
    if rzz_count!=1 or sorted(measured)!=[0,1]:
        raise ValueError('Expected one RZZ and a terminal two-bit readout')
    flush()
    return word
