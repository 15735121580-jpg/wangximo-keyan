"""Lower bounded polynomial feedback diagrams to the bundled exact and interval solvers."""

from dataclasses import asdict
from fractions import Fraction
import json
import sys


def rational(value):
    if type(value) not in (str, int):
        raise ValueError('Use an integer or a quoted rational or decimal.')
    return Fraction(value)


def bounds(pair):
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError('Expected lower and upper bounds.')
    low, high = map(rational, pair)
    if low > high:
        raise ValueError('Reversed bounds.')
    return low, high


def keys(record, required):
    if not isinstance(record, dict) or set(record) != set(required.split()):
        raise ValueError('Expected fields: ' + required)


def label(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Expected a nonempty name.')
    return name


def evaluate(expr, values, constant, binary):
    if not isinstance(expr, list):
        return constant(rational(expr))
    if len(expr) == 2 and expr[0] == 'var':
        if expr[1] not in values:
            raise ValueError('Undeclared control: ' + str(expr[1]))
        return values[expr[1]]
    if len(expr) != 3 or expr[0] not in ('+', '-', '*'):
        raise ValueError('Expected a constant, [var,name] or a binary +, -, * expression.')
    return binary(expr[0], evaluate(expr[1], values, constant, binary),
                  evaluate(expr[2], values, constant, binary))


class Policy:
    def __init__(self, spec, input_port, output_port):
        keys(spec, 'name controls root nodes residuals')
        self.name = label(spec['name'])
        if not isinstance(spec['controls'], dict) or not isinstance(spec['nodes'], list):
            raise ValueError('Expected a control map and node array.')
        self.controls = {label(name): bounds(pair) for name, pair in spec['controls'].items()}
        self.root = spec['root']
        self.nodes = {}
        for node in spec['nodes']:
            keys(node, 'id input output duration outcomes')
            name = label(node['id'])
            if name in self.nodes:
                raise ValueError('Repeated node ' + name)
            label(node['input'])
            label(node['output'])
            self.nodes[name] = node
        if self.root not in self.nodes or self.nodes[self.root]['input'] != input_port:
            raise ValueError('Root node or input port mismatch.')
        if not isinstance(spec['residuals'], list):
            raise ValueError('residuals must be an array.')
        self.residuals = spec['residuals']
        self.order = []
        self.edges = {}
        visited, active = set(), set()

        def visit(name):
            if name in active:
                raise ValueError('Feedback diagram contains an unbounded cycle.')
            if name in visited:
                return
            active.add(name)
            node = self.nodes[name]
            outcomes = node['outcomes']
            if not isinstance(outcomes, dict):
                raise ValueError('outcomes must map observable records to [probability, successor].')
            edges = []
            for observed, edge in outcomes.items():
                label(observed)
                if not isinstance(edge, list) or len(edge) != 2:
                    raise ValueError('Expected [probability, successor].')
                probability, target = rational(edge[0]), edge[1]
                if probability <= 0 or target not in self.nodes:
                    raise ValueError('Invalid outcome probability or successor.')
                if self.nodes[target]['input'] != node['output']:
                    raise ValueError('Incompatible ports at ' + name + ' -> ' + target)
                visit(target)
                edges.append((probability, target))
            if edges and sum(p for p, _ in edges) != 1:
                raise ValueError('Outcome probabilities must sum exactly to 1.')
            if not edges and node['output'] != output_port:
                raise ValueError('Terminal output does not match the task.')
            self.edges[name] = edges
            active.remove(name)
            visited.add(name)
            self.order.append(name)

        visit(self.root)
        if len(visited) != len(self.nodes):
            raise ValueError('Unreachable nodes in policy.')

    def costs(self, values, constant, binary):
        totals = {}
        residuals = [evaluate(g, values, constant, binary) for g in self.residuals]
        for name in self.order:
            cost = evaluate(self.nodes[name]['duration'], values, constant, binary)
            residuals.append(binary('-', constant(Fraction(0)), cost))
            for prob, target in self.edges[name]:
                cost = binary('+', cost, binary('*', constant(prob), totals[target]))
            totals[name] = cost
        return totals[self.root], tuple(residuals)


def exact(policies):
    import operator
    import optimal_compiler as sdk
    import z3

    def const(q):
        return z3.RealVal(str(q))

    def binary(op, left, right):
        return {'+': operator.add, '-': operator.sub, '*': operator.mul}[op](left, right)

    diagrams = []
    for policy in policies:
        values = {name: z3.Real(name) for name in policy.controls}
        guards = []
        for name, (low, high) in policy.controls.items():
            guards.extend([values[name] >= const(low), values[name] <= const(high)])
        cost, residuals = policy.costs(values, const, binary)
        guards.extend(g <= 0 for g in residuals)
        diagrams.append(sdk.Diagram(policy.name, tuple(values.values()), z3.And(guards),
                                    cost, policy.root))
    result = sdk.solve(diagrams)
    if not sdk.verify(diagrams, result):
        raise RuntimeError('Exact solver certificate failed verification.')
    return {'status': result.status, 'certificate_rechecked': True,
            'selected': result.diagram.name if result.diagram else None,
            'time': result.time.sexpr() if result.time is not None else None,
            'controls': result.exact_values()}


def interval(policies, tolerance, rounds):
    import interval_compiler as sdk

    def constant(q):
        return sdk.Interval(q, q)

    def binary(op, left, right):
        if op == '+':
            return sdk.Interval(left.lo + right.lo, left.hi + right.hi)
        if op == '-':
            return sdk.Interval(left.lo - right.hi, left.hi - right.lo)
        products = [a * b for a in (left.lo, left.hi) for b in (right.lo, right.hi)]
        return sdk.Interval(min(products), max(products))

    domains = []
    for policy in policies:
        box = tuple(sdk.Interval(low, high) for low, high in policy.controls.values())

        def enclose(box, precision, current=policy):
            values = dict(zip(current.controls, box))
            time, residuals = current.costs(values, constant, binary)
            return sdk.Evaluation(time, residuals)

        domains.append(sdk.Domain(policy.name, box, len(policy.residuals) + len(policy.nodes), enclose))
    if type(rounds) is not int or rounds < 0:
        raise ValueError('Specify a nonnegative integer max_rounds.')
    delta = rational(tolerance)
    result = sdk.solve(domains, delta=delta, max_rounds=rounds)
    if not sdk.verify(domains, delta, result):
        raise RuntimeError('Interval certificate failed verification.')
    return {'status': result.status, 'certificate_rechecked': True,
            'selected': result.witness.candidate.domain if result.witness else None,
            'certificate': asdict(result)}


def run_typed(job, skill, directory):
    mode = job['mode']
    fields = 'mode task input_port output_port time_unit policies'
    if mode == 'interval':
        fields += ' tolerance max_rounds'
    keys(job, fields)
    for field in ('task', 'input_port', 'output_port', 'time_unit'):
        label(job[field])
    if not isinstance(job['policies'], list) or not job['policies']:
        raise ValueError('Supply at least one policy.')
    policies = [Policy(p, job['input_port'], job['output_port']) for p in job['policies']]
    sys.path.insert(0, str(skill / 'assets/certified'))
    if mode == 'finite':
        receipt = exact(policies)
    else:
        receipt = interval(policies, job['tolerance'], job['max_rounds'])
    receipt.update(task=job['task'], time_unit=job['time_unit'], mode=mode)
    (directory / 'result.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    return 0 if receipt['status'] in ('optimal', 'delta_optimal') else 2
