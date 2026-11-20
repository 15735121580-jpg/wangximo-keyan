"""Two-shot VQE with an online, repeatable classical feedback barrier.

An exact model calculation. Between shots all quantum registers have been
measured and released. Classical polling uses wall-clock time but no quantum
depth. The basis update still consumes only the first public measurement.
"""
from fractions import Fraction as F
from cyclic_compiler import Action, Job, Outcome, compile_stopping_job
from feedback_vqe import leaf_distribution, symbolic_fixture


def make_job() -> Job:
    table, limits = symbolic_fixture()
    ideal = leaf_distribution(0)
    actual = leaf_distribution(table.flip)
    def label(pair):
        return ','.join(str(sign) for sign in pair)
    labels = {label(y): prob for y, prob in ideal.items()}
    first_probs = {i: sum(v for y, v in actual.items() if y[0] == i) for i in (1, -1)}
    actions = {}
    roots = []
    for layout in ('e', 's'):
        edges = tuple(Outcome(prob, next_state=f'barrier/{layout}/{i}') for i, prob in first_probs.items())
        elapsed = table.a + (table.delta if layout == 'e' else 0)
        roots.append(Action('encoded-A/'+layout, elapsed, table.peak_atoms, table.prefix_depth, edges))
        for first, prob in first_probs.items():
            barrier, run = f'barrier/{layout}/{first}', f'round-2/{layout}/{first}'
            actions[barrier] = (
                Action('poll', 1, 0, 0, (Outcome(F(1,2), next_state=run), Outcome(F(1,2), next_state=barrier))),
                Action('synchronous-result', 3, 0, 0, (Outcome(1, next_state=run),)))
            second_time = table.b
            if layout == 's' and first == 1:
                second_time += table.eta
            if layout == 'e' and first == -1:
                second_time += table.kappa
            outputs = []
            for second in (1, -1):
                pair = first, second
                # Marginalize the hidden first readout flip. There is no
                # policy-visible node for the no-fault/fault distinction.
                good = (1-table.flip)**2 * ideal[pair] / prob
                bad = actual[pair] / prob - good
                outputs.extend((Outcome(good, output=label(pair)),
                                Outcome(bad, output=label(pair), failed=True)))
            parameter = 'B' if first == 1 else 'A'
            actions[run] = (Action('encoded-'+parameter, second_time, table.peak_atoms,
                                   table.suffix_depth, tuple(outputs)),)
    actions['start'] = tuple(roots)
    return Job('start', actions, labels, limits.atoms, limits.depth, limits.error, limits.fidelity)


if __name__ == '__main__':
    solution = compile_stopping_job(make_job())
    stats = solution.statistics
    print('status', solution.result.status)
    if stats is not None:
        print('time', stats.expected_time, 'failure', stats.failure)
        print('first action', solution.policy[('start', 0)])
        print('reachable barriers', {s: p for s, p in solution.policy.items() if s[0].startswith('barrier/')})
        print('terminal output law', stats.output)
