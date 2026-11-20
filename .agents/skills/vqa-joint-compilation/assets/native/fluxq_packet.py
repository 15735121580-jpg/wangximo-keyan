"""Convert FluxQ's bound circuit from a Markdown intent to a vqa-joint packet.

Run inside the external FluxQ 0.2.0 Python 3.11 environment with Qiskit.
One binding represents one circuit call. The external VQA controller supplies
the other bindings, feedback and stopping rule for a full adaptive job.
"""
import argparse
import json
import math
from pathlib import Path
from quantum_runtime.intent.parser import parse_intent_file
from quantum_runtime.intent.planner import plan_to_qspec
from quantum_runtime.lowering.qiskit_emitter import build_qiskit_circuit


def as_packet(path):
    spec = plan_to_qspec(parse_intent_file(path))
    circuit = build_qiskit_circuit(spec)
    allowed = {'h','x','y','z','s','sdg','t','tdg','rx','ry','rz','rzz','cx','cz','measure'}
    n = circuit.num_qubits
    if not 1 <= n <= 8 or circuit.num_clbits != n or circuit.parameters:
        raise ValueError('Provide a bound small-register call with full readout')
    if (spec.constraints.max_width is not None and n > spec.constraints.max_width or
            spec.constraints.max_depth is not None and circuit.depth() > spec.constraints.max_depth):
        raise ValueError('Emitted source circuit exceeds its declared circuit bounds')
    if spec.constraints.connectivity_map:
        raise ValueError('Source-constrained connectivity requires its own matching target adapter')
    gates = []
    for inst in circuit.data:
        name = inst.operation.name
        q = [circuit.find_bit(w).index for w in inst.qubits]
        c = [circuit.find_bit(w).index for w in inst.clbits]
        if name not in allowed or (name == 'measure' and c != q) or (name != 'measure' and c):
            raise ValueError('Unsupported instruction or non-canonical classical mapping')
        a = list(map(float, inst.operation.params))
        if len(a) != int(name in {'rx','ry','rz','rzz'}) or not all(map(math.isfinite, a)):
            raise ValueError('Invalid gate angle')
        gates.append([name, q, a[0] if a else None])
    if gates[-n:] != [['measure',[i],None] for i in range(n)] or any(g[0]=='measure' for g in gates[:-n]):
        raise ValueError('This bridge requires terminal full-register measurement')

    # Restore a logical ZZ rotation from the emitted contiguous parity gadget.
    # The numeric angle comes from the actual emitted gate and retains its sign.
    out = []
    i = 0
    while i < len(gates):
        if i+2 < len(gates):
            left, center, right = gates[i:i+3]
            if (left[0] == 'cx' and right == left and center[0] == 'rz'
                    and center[1] == left[1][1:]):
                out.append(['rzz',left[1],center[2]])
                i += 3
                continue
        out.append(gates[i])
        i += 1
    shots=spec.constraints.shots
    if not isinstance(shots,int) or shots < 1:
        raise ValueError('The source must declare a positive integer shot count')
    return {'schema_version':1,'task_contract':spec.program_id,
            'stages':[{'id':'main','bindings':[{'id':'bound','width':n,'shots':shots,'operations':out}]}]}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('intent',type=Path)
    cli.add_argument('packet',type=Path)
    args=cli.parse_args()
    data=as_packet(args.intent)
    with args.packet.open('x',encoding='utf-8') as f:
        json.dump(data,f,indent=2); f.write('\n')
