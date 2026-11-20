"""Re-read the receipt, recompute the declared model, and check its witness."""
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path
from random import Random

import z3

root = Path('.agents/skills/vqa-joint-compilation/assets/certified').resolve()
sys.path.insert(0, str(root))
from optimal_compiler import Diagram, Result, verify
from feedback_vqe import Program

packet = json.loads(Path('contract.json').read_text())
receipt = json.loads(Path('result.json').read_text())
bounds = packet['limits']
diagrams = []
metrics = {}
for candidate in packet['architectures']:
    name = candidate['id']
    r = F(candidate['flip_per_readout'])
    p = F(3, 4) * (1-r) + F(1, 4) * r
    noisy = {(1, 1): p*(1-p), (1, -1): p*p,
             (-1, 1): (1-p)*p, (-1, -1): (1-p)*(1-p)}
    ideal = {(1, 1): F(3,16), (1,-1): F(9,16),
             (-1,1): F(3,16), (-1,-1): F(1,16)}
    affinity = sum(z3.Sqrt(z3.RealVal(str(ideal[h]*noisy[h]))) for h in ideal)
    fidelity_ok = z3.is_true(z3.simplify(affinity**2 >= z3.RealVal(bounds['min_record_fidelity'])))
    fidelity = sum(math.sqrt(float(ideal[h]*noisy[h])) for h in ideal)**2
    time = F(candidate['prefix_us']) + p*F(candidate['suffix_us']['+']) + (1-p)*F(candidate['suffix_us']['-'])
    flip = 1-(1-r)**2
    checks = dict(peak_atoms=sum(candidate['peak_atoms_by_species'].values()) <= bounds['max_peak_atoms'],
                  branch_depth=candidate['max_branch_depth'] <= bounds['max_depth'],
                  model_flip_probability=flip <= F(bounds['max_model_flip_probability']),
                  record_fidelity=fidelity_ok)
    recorded = receipt['architectures'][name]
    assert checks == recorded['constraints_met']
    assert str(time) == recorded['expected_us_exact']
    assert str(flip) == recorded['p_any_model_readout_flip_exact']
    assert math.isclose(fidelity, recorded['record_fidelity_approx'], abs_tol=1e-15)
    diagrams.append(Diagram(name, (), z3.BoolVal(all(checks.values())), z3.RealVal(str(time)), name))
    metrics[name] = {'expected_us_exact': str(time), 'expected_us': float(time),
                     'positive_first_readout_probability_exact': str(p)}

winner = next(d for d in diagrams if d.name == receipt['selected'])
assert verify(diagrams, Result('optimal', winner.time, winner))
assert receipt['certificate_rechecked'] is True
ties = [d.name for d in diagrams if z3.is_true(z3.simplify(z3.And(d.feasible, d.time == winner.time)))]
assert ties == receipt['ties']
seed = packet['task_semantics']['replay_seed']
scale = math.lcm(*(v.denominator for v in noisy.values()))
draw = Random(seed).randrange(scale)
remaining = draw
for pair, weight in noisy.items():
    remaining -= int(weight * scale)
    if remaining < 0:
        break
replayed = Program('e').replay(*pair)
actual = receipt['seeded_model_replay']
assert actual['seed'] == seed
assert actual['readouts'] == list(pair)
assert actual['parameters'] == list(replayed['parameters'])
assert actual['choice'] == replayed['choice']
assert actual['empirical_energies_exact'] == {k: str(v) for k,v in replayed['empirical_energies'].items()}
selected_input = next(c for c in packet['architectures'] if c['id'] == winner.name)
actual_time = F(selected_input['prefix_us']) + F(selected_input['suffix_us']['+' if pair[0] == 1 else '-'])
audit = {
    'receipt_matches_contract': True,
    'independent_optimal_certificate_verified': True,
    'seeded_replay_verified': True,
    'candidate_metrics': metrics,
    'selected': winner.name,
    'time_saving_us_exact': str(F(metrics['forward-chain']['expected_us_exact']) - F(metrics['branch-local']['expected_us_exact'])),
    'record_distribution_exact': {str(h): str(v) for h,v in noisy.items()},
    'integer_sampling_scale': scale,
    'integer_sampling_draw': draw,
    'actual_readouts': list(pair),
    'actual_parameters': list(replayed['parameters']),
    'actual_choice': replayed['choice'],
    'actual_empirical_energies_exact': actual['empirical_energies_exact'],
    'replayed_branch_duration_us_exact': str(actual_time),
    'scope': 'Local synthetic model; finite declared catalogue only'
}
Path('verification.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit, indent=2))
