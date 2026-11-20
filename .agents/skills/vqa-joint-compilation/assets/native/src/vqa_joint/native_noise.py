"""Bound native U/CZ gate instruments under an explicit memoryless noise model."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from vqa_joint.contract import DIGITAL_GATE_INPUTS

AVERAGE_FIDELITY = {i: p.low for i, (_, p) in enumerate(DIGITAL_GATE_INPUTS, 1)}


def local_unitary(gate):
    name=gate['gate']
    args=gate.get('params',[])
    if name=='cz' and len(args)==0:
        return np.diag([1.,1.,1.,-1.])
    if name in ('u','u3'):
        t,a,b=args
    elif name=='u2':
        a,b=args
        t=np.pi/2
    elif name in ('u1','p'):
        b,=args
        t=a=0.
    elif name=='id' and not args:
        return np.eye(2)
    else:
        raise ValueError(f'Unsupported bound native gate {name}')
    c,s=np.cos(t/2),np.sin(t/2)
    return np.array([[c,-np.exp(1j*b)*s],[np.exp(1j*a)*s,np.exp(1j*(a+b))*c]])


def evolve_density(rho, n, gates, noise):
    """Apply native gates to a density matrix that may also contain an untouched reference."""
    if noise not in ('ideal_gates','local_depolarizing'):
        raise ValueError('Unknown gate channel')
    d=1<<n
    for gate in gates:
        local=local_unitary(gate)
        wires=tuple(gate['qubits'])
        k=len(wires)
        if (1<<k)!=len(local) or len(set(wires))!=k or any(i not in range(n) for i in wires):
            raise ValueError('Native gate port mismatch')
        # Convert little-endian qubit numbers to row/column tensor axes.
        acted=[n-1-i for i in reversed(wires)]
        other=[a for a in range(n) if a not in acted]
        axes=other+acted+[n+a for a in other+acted]
        L=len(local)
        blocks=rho.reshape([2]*(2*n)).transpose(axes).reshape(d//L,L,d//L,L)
        blocks=np.einsum('is,asbt,jt->aibj',local,blocks,local.conj(),optimize=True)
        if noise=='local_depolarizing':
            mix=(1-AVERAGE_FIDELITY[k])*L/(L-1)
            reduced=np.trace(blocks,axis1=1,axis2=3)
            blocks=(1-mix)*blocks+mix*np.einsum('ab,ij->aibj',reduced,np.eye(L)/L)
        rho=blocks.reshape([2]*(2*n)).transpose(np.argsort(axes)).reshape(d,d)
    return rho


@lru_cache(maxsize=512)
def _law(serialized, noise, readout_error):
    data=json.loads(serialized)
    n=data['num_qubits']
    if not 1 <= n <= 8 or not 0<=readout_error<=1:
        raise ValueError('Unsupported register width or readout probability')
    d=1<<n
    rho=np.zeros((d,d),complex)
    rho[0,0]=1.
    rho=evolve_density(rho,n,data['gates'],noise)
    p=np.diag(rho).real.copy()
    if p.min() < -1e-11 or abs(p.sum()-1)>1e-10:
        raise ArithmeticError('Native channel did not preserve probabilities')
    p=np.maximum(p,0)
    if readout_error:
        indexes=np.arange(d)
        for q in range(n):
            p=(1-readout_error)*p+readout_error*p[indexes^(1<<q)]
    return tuple(p/p.sum())


def outcome_law(path, noise, readout_error):
    return _law(Path(path).read_text(),noise,readout_error)
