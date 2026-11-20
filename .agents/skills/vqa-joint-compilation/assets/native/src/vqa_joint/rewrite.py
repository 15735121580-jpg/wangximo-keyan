from __future__ import annotations
"""Cross-pass handling of parity frames between logical operations and host routing.

A diagonal phase is tagged by its original input parity. Synthesis may move
or share parity computations but must close the exact outgoing wire map.
"""
from collections import Counter
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Op:
    name: str
    wires: tuple
    angle: float | None = None


def normalize(circuit):
    result = []
    ended = False
    for name, wires, angle in circuit:
        name = name.lower()
        if name == "measure":
            ended = True
            continue
        if ended:
            raise ValueError("A mid-circuit measurement requires a feedback-aware boundary")
        result.append(Op(name, tuple(wires), angle))
    return result


def standard(block):
    out = []
    for op in block:
        if op.name == "rzz":
            a,b = op.wires
            out.extend((Op("cx",(a,b)),Op("rz",(b,),op.angle),Op("cx",(a,b))))
        else:
            out.append(op)
    return out


def signature(block, n):
    rows = [1 << i for i in range(n)]
    phases = []
    for op in block:
        if op.name == "cx":
            a,b = op.wires
            rows[b] ^= rows[a]
        elif op.name == "rz":
            phases.append((rows[op.wires[0]],op.angle))
        else:
            raise ValueError(op.name)
    return phases, rows


def coordinates(rows, mask):
    # Echelon basis carries the coefficients in the current physical rows.
    basis = {}
    for i, row in enumerate(rows):
        x, coeff = row, 1 << i
        while x:
            lead = x.bit_length()-1
            if lead not in basis:
                basis[lead] = x,coeff
                break
            v,c = basis[lead]
            x ^= v
            coeff ^= c
    coeff = 0
    while mask:
        v,c = basis[mask.bit_length()-1]
        mask ^= v
        coeff ^= c
    return coeff


def close_frame(rows, target):
    matrix = [coordinates(rows, row) for row in target]
    n = len(matrix)
    steps = []
    def add(a,b):
        matrix[b] ^= matrix[a]
        steps.append(Op("cx",(a,b)))
    for i in range(n):
        if not matrix[i] & (1 << i):
            j = next(j for j in range(i+1,n) if matrix[j] & (1 << i))
            add(j,i)
        for j in range(n):
            if j != i and matrix[j] & (1 << i):
                add(i,j)
    return list(reversed(steps))


def synthesize(block, n, preference):
    original = standard(block)
    phases, wanted = signature(original,n)
    current = [1 << i for i in range(n)]
    pending = list(enumerate(phases))
    out = []
    while pending:
        choices = []
        for j,(idx,(mask,angle)) in enumerate(pending):
            coeff = coordinates(current,mask)
            active = [w for w in range(n) if coeff & (1 << w)]
            # Deterministic port choices leave distinct layouts for native routing.
            for pivot in active:
                weight = sum(abs(pivot-w) for w in active) if preference == 'near' else 0
                direction = -pivot if preference == 'reverse' else pivot
                choices.append(((len(active),weight,direction,idx),j,pivot,active))
        _,j,pivot,active = min(choices)
        _,(_,angle) = pending.pop(j)
        for w in active:
            if w != pivot:
                out.append(Op("cx",(w,pivot)))
                current[pivot] ^= current[w]
        out.append(Op("rz",(pivot,),angle))
    out.extend(close_frame(current,wanted))
    got, actual = signature(out,n)
    if actual != wanted or Counter(got) != Counter(phases):
        raise ArithmeticError("Port map or logical phase mismatch")
    return out


def variants(circuit, n):
    """Return identity and certified full-call alternatives at closed boundaries."""
    src = normalize(circuit)
    if any(not op.wires or len(set(op.wires))!=len(op.wires) or
           any(q<0 or q>=n for q in op.wires) for op in src):
        raise ValueError("Invalid wires")
    generated = {}
    for mode in ('original','near','forward','reverse'):
        out, block = [], []
        def flush():
            if block:
                out.extend(standard(block) if mode == 'original' else synthesize(block,n,mode))
                block.clear()
        for op in src:
            if op.name in ('rz','rzz','cx'):
                block.append(op)
            else:
                flush()
                out.append(op)
        flush()
        if out not in generated.values():
            generated[mode] = out
    return generated


def qasm(ops,n):
    lines=['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{n}];']
    for op in ops:
        if op.name not in ('cx','h','x','y','z','cz','s','sdg','t','tdg','rx','ry','rz'):
            raise ValueError(op.name)
        args=','.join(f'q[{w}]' for w in op.wires)
        is_rotation = op.name in ('rx','ry','rz')
        if is_rotation != (op.angle is not None) or is_rotation and not math.isfinite(op.angle):
            raise ValueError("Invalid bound angle")
        param=f'({float(op.angle):.17g})' if is_rotation else ''
        lines.append(f'{op.name}{param} {args};')
    return '\n'.join(lines)+'\n'
