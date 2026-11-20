"""Compile a three-mode, one-particle diagonal-cost XY program using the installed SDK."""
import json

from qiskit import QuantumCircuit, qasm2, transpile
from vqa_joint.sector_qaoa.fixed_weight import CostMixerLayer, FixedWeightPort


def main():
    port = FixedWeightPort(3, 1, seed_mask=2)
    prep = ((0, 1, .31),)
    layer = CostMixerLayer(.7, lambda mask: mask & 1, ((1, 2, -.4), (0, 2, .16)))
    source = QuantumCircuit(3)
    source.x(1)

    def exchange(pair):
        i, j, angle = pair
        source.rxx(angle, i, j)
        source.ryy(angle, i, j)

    for pair in prep:
        exchange(pair)
    source.p(-layer.gamma, 0)
    for pair in layer.exchanges:
        exchange(pair)
    native = transpile(source, basis_gates=['u3', 'cz'], optimization_level=1, seed_transpiler=0)
    options = port.implementations(qasm2.dumps(native), port.bind(prep, (layer,)))
    report = {key: dict(width=plan.width, readout=plan.readout, residual=plan.verified_residual)
              for key, plan in options.items()}
    assert report['joint']['readout'] == (1, 2, 4, None)
    assert all(x['residual'] < 1e-10 for x in report.values())
    print(json.dumps(report))


if __name__ == '__main__':
    main()
