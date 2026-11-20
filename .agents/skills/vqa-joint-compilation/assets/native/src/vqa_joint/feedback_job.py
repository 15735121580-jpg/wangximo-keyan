"""Connect a finite classical feedback graph to compiled host instruments."""
from functools import lru_cache
from math import isfinite
from pathlib import Path

import numpy as np
from vqa_joint.feedback import FeedbackProgram, Observation
from vqa_joint.coherent_port import fidelity, source_target
from vqa_joint.native_noise import outcome_law
from vqa_joint.ports import WirePort
from vqa_joint.sector_dispatch import Readout


CLOCK_TERMS=('preparation_us_per_shot','readout_us_per_shot','random_us_per_bit',
             'decision_us_per_branch','processing_us_per_batch','finalization_us')
FEEDBACK_COSTS=('duration_us','gate_count','peak_data_atoms','gate_dependency_depth',
                'task_failure_probability','modeled_job_time_us','max_circuit_process_infidelity')


class FeedbackJob:
    def __init__(self, spec, stages):
        self.graph=FeedbackProgram(spec['entry'],spec['nodes'])
        active=set()
        for node in self.graph.nodes.values():
            if node['kind']=='batch':
                stage,key=node['stage'],node['binding']
                if stage not in stages or key not in stages[stage]:
                    raise ValueError('Feedback graph references an unknown circuit')
                width=stages[stage][key][0]
                if len(node['scores'])!=1<<width:
                    raise ValueError('Score table must include every original logical outcome')
                active.add((stage,key))
        if active!={(s,k) for s,r in stages.items() for k in r}:
            raise ValueError('Every declared binding must belong to the feedback graph')
        self.noise=spec['gate_noise']
        self.readout_error=spec['readout_flip_probability']
        if self.noise not in ('ideal_gates','local_depolarizing') or not 0<=self.readout_error<=1:
            raise ValueError('Unknown gate noise model or invalid readout flip probability')
        self.clock=spec.get('clock_us')
        if self.clock is not None and (set(self.clock)!=set(CLOCK_TERMS) or
                any(type(x) not in (int,float) or not isfinite(x) or x<0 for x in self.clock.values())):
            raise ValueError('Supply all six finite nonnegative serial time terms')
        self.floor=spec.get('minimum_task_success',0.)
        if not 0<=self.floor<=1:
            raise ValueError('Invalid minimum task success')
        self.fidelity_floor=spec.get('minimum_circuit_process_fidelity')
        self.coherent_port=spec.get('circuit_process_port')
        if self.fidelity_floor is not None and (self.coherent_port is None or not 0 <= self.fidelity_floor <= 1):
            raise ValueError('A fidelity threshold needs a recognized common circuit process port')
        self.spaces=({(stage,key):source_target(ops,self.coherent_port)
                     for stage,bindings in stages.items() for key,(_,ops) in bindings.items()}
                     if self.coherent_port else {})
        self.fidelities={}

    def assess(self, selected):
        views={}
        def observe(stage,key):
            if (stage,key) not in views:
                c=selected[stage][key]
                m=c.lowered.native_metrics
                native_law=outcome_law(c.lowered.schedule,self.noise,self.readout_error)
                law=c.logical_probabilities(native_law)
                cost={k:m[k] for k in ('duration_us','gate_count','gate_dependency_depth')}
                cost['random_bits']=sum(prob*c.readout.mean_random_bits(c.lowered.wire_port.logical_outcome(i))
                                        for i,prob in enumerate(native_law))
                views[stage,key]=Observation(tuple(law),m['n_data_atoms'],cost)
            return views[stage,key]
        report=self.graph.evaluate(observe)
        e=report['expected']
        cost={k:e[k] for k in ('duration_us','gate_count','gate_dependency_depth')}
        cost.update(peak_data_atoms=report['peak_atoms'],success_probability=report['success_probability'],
                    task_failure_probability=report['failure_probability'])
        if self.spaces:
            fs={}
            for (stage,key),space in self.spaces.items():
                call=selected[stage][key]
                cache_key=(stage,key,call.family,call.variant,call.lowered.schedule)
                if cache_key not in self.fidelities:
                    self.fidelities[cache_key]=fidelity(call,space,self.noise)
                fs.setdefault(stage,{})[key]=self.fidelities[cache_key]
            cost['max_circuit_process_infidelity']=max(1-f for row in fs.values() for f in row.values())
            report['circuit_process_fidelity_by_binding']=fs
        if self.clock is not None:
            t=self.clock
            cost['modeled_job_time_us']=(cost['duration_us']
                +e['shots']*(t['preparation_us_per_shot']+t['readout_us_per_shot'])
                +e.get('random_bits',0)*t['random_us_per_bit']
                +e.get('branch_tests',0)*t['decision_us_per_branch']
                +e['batches']*t['processing_us_per_batch']+t['finalization_us'])
        return cost,report


def sample_saved(result, root, seed):
    spec=result['feedback']
    graph=FeedbackProgram(spec['entry'],spec['nodes'])
    root=Path(root).resolve()
    rng=np.random.default_rng(seed)
    @lru_cache(maxsize=None)
    def instrument(stage,key):
        c=result['selected'][stage][key]
        path=(root/c['files']['bound_schedule.json']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Native schedule must be inside this result')
        return (outcome_law(path,spec['gate_noise'],spec['readout_flip_probability']),
                WirePort(tuple(c['wire_order'])),Readout(**c['readout']))
    def draw(stage,key,shots):
        p,port,kernel=instrument(stage,key)
        for sample in rng.choice(len(p),size=shots,p=p):
            slot=port.logical_outcome(int(sample))
            index=kernel.sample_index(slot,lambda bits:int(rng.integers(1<<bits)))
            yield kernel.decode_bits(slot,index)
    return dict(seed=seed,gate_noise=spec['gate_noise'],**graph.run(draw))
