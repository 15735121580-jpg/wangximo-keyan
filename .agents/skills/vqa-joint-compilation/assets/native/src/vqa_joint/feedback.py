"""Finite adaptive job on decoded logical outcome distributions.

At a batch, shots are independent fresh preparations under the supplied model.
An incoming state retains only the classical register values the suffix reads.
This evaluates the stochastic controller without sampling its training branches.
"""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from math import fsum, isclose, isfinite
from operator import eq, ge, gt, le, lt

import numpy as np


COMPARE = {'==':eq, '>=':ge, '>':gt, '<=':le, '<':lt}


@dataclass(frozen=True)
class Observation:
    logical_probabilities: tuple
    data_atoms: int
    per_shot_costs: dict


def _batch_distribution(probabilities, scores, shots, reduction):
    p = np.asarray(probabilities, dtype=float)
    if len(p) != len(scores) or p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError('Invalid decoded outcome distribution')
    if not isclose(fsum(p), 1, rel_tol=0, abs_tol=1e-10):
        raise ValueError('Decoded outcomes must exhaust the shot probability')
    p = p / fsum(p)
    one = Counter()
    for score, prob in zip(scores,p):
        if type(score) is not int:
            raise ValueError('A finite score must be an integer')
        if prob:
            one[score] += prob
    if reduction == 'max':
        keys = sorted(one)
        cdf = np.cumsum([one[k] for k in keys])
        cdf[-1] = 1
        values = np.diff(np.r_[0.0, np.clip(cdf,0,1) ** shots])
        return [(k,float(v)) for k,v in zip(keys,values) if v]
    if reduction != 'sum':
        raise ValueError('Supported shot reductions are sum and max')
    low,high=min(one),max(one)
    if low == high:
        return [(low*shots,1.0)]
    # Sparse integer score grids are compressed before convolving.
    stride = int(np.gcd.reduce([k-low for k in one]))
    length = (high-low)//stride
    if length*shots > 2_000_000:
        raise ValueError('Score distribution exceeds the dense convolution budget')
    base = np.zeros(length+1)
    for k,mass in one.items():
        base[(k-low)//stride] = mass
    dist = np.array([1.0])
    remaining = shots
    while remaining:
        if remaining & 1:
            dist = np.convolve(dist,base)
        remaining >>= 1
        if remaining:
            base = np.convolve(base,base)
    return [(low*shots + int(i)*stride, float(dist[i])) for i in np.flatnonzero(dist)]


class FeedbackProgram:
    """A finite DAG of measurement batches, integer comparisons and returns."""

    def __init__(self, entry, nodes):
        self.entry = entry
        self.nodes = dict(nodes)
        if not self.nodes or entry not in self.nodes:
            raise ValueError('Missing feedback entry')
        edges, reads, writes = {}, {}, {}
        for name,node in self.nodes.items():
            kind = node['kind']
            if kind == 'batch':
                if type(node['shots']) is not int or node['shots'] < 1 or not node['scores']:
                    raise ValueError('Every batch needs positive shots and outcome scores')
                if node.get('reduce','sum') not in ('sum','max') or any(type(s) is not int for s in node['scores']):
                    raise ValueError('Expected sum/max over integer scores')
                edges[name], reads[name], writes[name] = [node['next']], set(node.get('add',[])), {node['save']}
            elif kind == 'branch':
                if node['op'] not in COMPARE:
                    raise ValueError('Unknown comparison')
                operands=[node['left'],node['right']]
                if any(type(x) not in (str,int) for x in operands):
                    raise ValueError('Compare named registers or integer literals')
                edges[name],reads[name],writes[name]=[node['yes'],node['no']],{x for x in operands if isinstance(x,str)},set()
            elif kind == 'finish' and type(node['success']) is bool:
                edges[name],reads[name],writes[name]=[],({node['return']} if 'return' in node else set()),set()
            else:
                raise ValueError('Unknown feedback operation')
            if not all(x in self.nodes for x in edges[name]):
                raise ValueError('Missing feedback continuation')
        incoming = Counter(child for children in edges.values() for child in children)
        queue = deque([n for n in self.nodes if not incoming[n]])
        if list(queue) != [entry]:
            raise ValueError('Feedback job requires one reachable entry')
        self.order=[]
        while queue:
            name=queue.popleft()
            self.order.append(name)
            for child in edges[name]:
                incoming[child]-=1
                if incoming[child] == 0:
                    queue.append(child)
        if len(self.order)!=len(self.nodes):
            raise ValueError('Feedback job must terminate on every branch')
        self.live={}
        for name in reversed(self.order):
            later=set().union(*(self.live[child] for child in edges[name]))
            self.live[name]=tuple(sorted(reads[name] | (later - writes[name])))
        if self.live[entry]:
            raise ValueError(f'Uninitialized feedback registers: {self.live[entry]}')

    def evaluate(self, observe):
        """observe(stage,binding) supplies the native instrument and per-shot costs."""
        queues = {name:defaultdict(float) for name in self.order}
        queues[self.entry][()] = 1.
        totals, answers, returns, visits = Counter(),Counter(),Counter(),{}
        peak = 0
        cost_keys = None
        for name in self.order:
            population = queues.pop(name)
            mass = fsum(population.values())
            visits[name]=mass
            if mass == 0:
                continue
            node=self.nodes[name]
            def emit(child, regs, weight):
                if weight <= 0:
                    return
                key=tuple(regs[k] for k in self.live[child])
                queues[child][key]+=weight
            kind=node['kind']
            if kind == 'batch':
                view=observe(node['stage'],node['binding'])
                if cost_keys is None:
                    cost_keys=set(view.per_shot_costs)
                    if cost_keys & {'shots','batches','branch_tests'}:
                        raise ValueError('Per-shot costs overlap controller counters')
                if set(view.per_shot_costs)!=cost_keys or any(not isfinite(x) or x < 0 for x in view.per_shot_costs.values()):
                    raise ValueError('Every reachable batch must supply the same finite nonnegative costs')
                if type(view.data_atoms) is not int or view.data_atoms<0:
                    raise ValueError('Invalid atom count')
                law=_batch_distribution(view.logical_probabilities,node['scores'],node['shots'],node.get('reduce','sum'))
                if not isclose(fsum(x for _,x in law),1.,rel_tol=0,abs_tol=5e-10):
                    raise ArithmeticError('Batch law lost probability mass')
                peak=max(peak,view.data_atoms)
                for k,v in view.per_shot_costs.items():
                    totals[k]+=mass*node['shots']*v
                totals['shots']+=mass*node['shots']
                totals['batches']+=mass
            elif kind == 'branch':
                totals['branch_tests']+=mass
            for key,weight in population.items():
                regs=dict(zip(self.live[name],key))
                if kind=='batch':
                    offset=sum(regs[x] for x in node.get('add',[]))
                    for score,prob in law:
                        regs[node['save']] = score+offset
                        emit(node['next'],regs,weight*prob)
                elif kind=='branch':
                    left,right=(regs[x] if isinstance(x,str) else x for x in (node['left'],node['right']))
                    emit(node['yes'] if COMPARE[node['op']](left,right) else node['no'],regs,weight)
                else:
                    returns[node['success']]+=weight
                    answers[(name,regs.get(node.get('return')))]+=weight
        terminal_mass = fsum(returns.values())
        if not isclose(terminal_mass,1.,rel_tol=0,abs_tol=5e-10):
            raise ArithmeticError('Feedback evaluation lost terminal probability')
        return dict(success_probability=returns[True]/terminal_mass,
                    failure_probability=returns[False]/terminal_mass,expected=dict(totals),peak_atoms=peak,
                    visited=visits,terminal=[dict(node=k[0],value=k[1],probability=v) for k,v in answers.items()])

    def run(self, draw):
        """Execute one job; draw(stage,binding,shots) supplies decoded wire labels."""
        name=self.entry
        regs={}
        trace=[]
        while True:
            node=self.nodes[name]
            kind=node['kind']
            if kind=='finish':
                return dict(terminal=name,success=node['success'],value=regs.get(node.get('return')),trace=trace)
            if kind=='branch':
                left,right=(regs[x] if isinstance(x,str) else x for x in (node['left'],node['right']))
                name=node['yes'] if COMPARE[node['op']](left,right) else node['no']
                continue
            labels=list(draw(node['stage'],node['binding'],node['shots']))
            if len(labels)!=node['shots'] or any(not isinstance(x,(int,np.integer)) or not 0<=x<len(node['scores']) for x in labels):
                raise ValueError('Device returned the wrong number or range of decoded outcomes')
            values=[node['scores'][x] for x in labels]
            reducer=sum if node.get('reduce','sum')=='sum' else max
            score=reducer(values)+sum(regs[x] for x in node.get('add',[]))
            regs[node['save']]=score
            trace.append(dict(node=name,stage=node['stage'],binding=node['binding'],shots=len(labels),score=score))
            name=node['next']
