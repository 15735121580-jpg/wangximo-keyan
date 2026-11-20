"""Extract and replay a finite certified-port policy for the six ZAC Steane VQE calls."""
from functools import lru_cache
from itertools import product
import json
from math import comb, fsum, isfinite, prod
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
T = (0, 1, -1)
B = ('X','Z')
KEYS = tuple(f'{t}/{b}' for t in T for b in B)
ARMS = ('verified','terminal')
MIN = {'min_application_success':'application_success',
       'min_answer_distribution_fidelity':'answer_distribution_fidelity',
       'min_preparation_fidelity':'preparation_fidelity',
       'min_circuit_process_fidelity':'circuit_process_fidelity',
       'min_native_gate_process_fidelity':'native_gate_process_lower'}
MAX = {'max_peak_atoms':'peak_atoms','max_expected_gates':'expected_gates',
       'max_expected_depth':'expected_depth','max_expected_time_ms':'expected_time_ms',
       'max_decoded_flip_fraction':'decoded_flip_fraction',
       'max_coupled_job_flip_probability':'coupled_job_flip_probability',
       'max_round_logical_error':'round_logical_error'}
SCOPES = {
    'application_success':'Select the grid minimizer and deliver a fresh energy within 0.20 of -1.',
    'preparation_fidelity':'Worst fresh-input, pre-readout seven-data-atom state. Marginalize the verifier before heralding.',
    'native_gate_process_lower':'Worst physical bound gate sequence relative to its own ideal unitary on its full physical input space. Excludes reset and readout.',
    'circuit_process_fidelity':'No shared arbitrary-input logical circuit channel for these heralded, fresh-preparation readout programs.',
    'round_logical_error':'No repeated non-destructive correction-round calibration at this task port.',
    'decoded_flip_fraction':'Expected number of accepted wrong coupled logical signs / expected number of accepted signs.',
    'coupled_job_flip_probability':'Probability at least one accepted sign differs from the coupled ideal sign in the feedback job.',
    'answer_distribution_fidelity':'Squared classical fidelity of (selected theta, fresh reported energy) to its full ideal law.',
    'resources':'Serial expected totals over training, conditional third evaluation, fresh validation and geometric retries. Peak atoms count concurrent resources.',
}
CHECKS = [[q for q in range(7) if (q+1)&(1<<bit)] for bit in range(3)]


def canonical(packet):
    if not isinstance(packet,dict) or set(packet)-{'host','task','train_shots_per_basis','constraints'}:
        raise ValueError('Expected a Steane task packet with recognized fields.')
    if packet.get('host')!='ZAC' or packet.get('task')!='steane_grid_vqe':
        raise ValueError('The encoded portfolio supports ZAC steane_grid_vqe.')
    n=packet.get('train_shots_per_basis',64)
    if type(n) is not int or n not in (16,64):
        raise ValueError('train_shots_per_basis must equal 16 or 64.')
    cs=packet.get('constraints',{})
    if not isinstance(cs,dict) or set(cs)-(set(MIN)|set(MAX)|{'require_verified_preparation'}):
        raise ValueError('Unknown constraint for the encoded portfolio.')
    for k,v in cs.items():
        if v is None:
            continue
        if k=='require_verified_preparation':
            if type(v) is not bool:
                raise ValueError(f'{k} must be boolean.')
        elif type(v) not in (int,float) or not isfinite(v) or v<0:
            raise ValueError(f'{k} must be finite and nonnegative.')
        elif (k in MIN or k in ('max_round_logical_error','max_coupled_job_flip_probability','max_decoded_flip_fraction')) and v>1:
            raise ValueError(f'{k} is a probability.')
        elif k=='max_peak_atoms' and type(v) is not int:
            raise ValueError(f'{k} must be an integer.')
    return dict(host='ZAC',task='steane_grid_vqe',train_shots_per_basis=n,
                constraints={k:v for k,v in cs.items() if v is not None})


@lru_cache(maxsize=100)
def distribution(n,px,pz):
    """Index is energy numerator + 7*n; the common energy denominator is 4*n."""
    def law(p):
        return np.array([comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)])
    k=np.arange(n+1)
    indexes=14*n-8*k[:,None]-6*k[None,:]
    return np.bincount(indexes.ravel(),weights=np.outer(law(px),law(pz)).ravel(),minlength=14*n+1)


def feedback(train):
    zero,plus,minus=train
    lt=lambda ps: np.r_[0.,np.cumsum(ps[:-1])]
    ge=lambda ps: np.cumsum(ps[::-1])[::-1]
    # If + wins, stop. Otherwise compare - against 0 using the SAME score for 0.
    return (float(zero@(ge(plus)*ge(minus))),float(zero@lt(plus)),float(zero@(ge(plus)*lt(minus))))


def assess(rows,n,clocks):
    valid=64
    def laws(m,ideal):
        def plus(r): return (1+r['ideal_pauli_expectation'])/2 if ideal else r['p_plus']
        return [distribution(m,plus(rows[i]),plus(rows[i+1])) for i in (0,2,4)]
    pr=feedback(laws(n,False))
    ref=feedback(laws(n,True))
    fresh=laws(valid,False)
    fresh_ref=laws(valid,True)
    # Integer check on the exact rational tolerance 1/5 Hartree.
    good=5*abs(np.arange(14*valid+1)-7*valid+4*valid)<=4*valid
    answer_fid=fsum((a*b)**.5*float(np.sqrt(u*v).sum()) for a,b,u,v in zip(pr,ref,fresh,fresh_ref))**2
    batches=3+pr[0]+pr[2]
    accepted=[n*(1 if j<2 else (pr[0]+pr[2]))+valid*pr[j] for j in range(3) for b in B]
    tries=[number/r['acceptance_probability'] for number,r in zip(accepted,rows)]
    def expected(field):
        return fsum(number*r[field] for number,r in zip(tries,rows))
    retries=fsum(tries)-fsum(accepted)
    overhead=sum(clocks[k] for k in ('fresh_reset_us_per_shot','parallel_image_us_per_shot','decode_us_per_shot'))
    flip=[r['logical_flip_probability'] for r in rows]
    def clean(j,m):
        return prod((1-flip[2*j+b])**m for b in (0,1))
    clean_job=clean(0,n)*clean(1,n)*fsum(ref[j]*clean(j,valid)*(1 if j==1 else clean(2,n)) for j in range(3))
    return dict(application_success=pr[1]*float(fresh[1][good].sum()),
                answer_distribution_fidelity=min(1.,answer_fid),peak_atoms=max(r['peak_atoms'] for r in rows),
                expected_gates=expected('gate_count'),expected_depth=expected('depth'),
                expected_time_ms=(expected('unitary_us')+overhead*fsum(tries)+clocks['controller_us_per_energy_batch']*batches)/1000,
                expected_attempts=fsum(tries),expected_rejected_attempts=retries,expected_energy_batches=batches,
                decoded_flip_fraction=fsum(n*p for n,p in zip(accepted,flip))/fsum(accepted),
                coupled_job_flip_probability=1-clean_job, round_logical_error=None, circuit_process_fidelity=None,
                preparation_fidelity=min(r['preparation_fidelity'] for r in rows),
                native_gate_process_lower=min(r['native_gate_process_lower'] for r in rows),
                native_gate_process_upper=min(r['native_gate_process_upper'] for r in rows))


def check(assignment,metrics,cs):
    failed=[];unknown=[]
    for key,limit in cs.items():
        if key=='require_verified_preparation':
            if limit and any(arm!='verified' for arm in assignment):
                failed.append(key)
            continue
        value=metrics[(MIN|MAX)[key]]
        if value is None:
            unknown.append(key)
        elif key=='min_native_gate_process_fidelity' and value<limit<=metrics['native_gate_process_upper']:
            unknown.append(key)
        elif (key in MIN and value<limit) or (key in MAX and value>limit):
            failed.append(key)
    return failed,unknown


def select_policy(packet):
    request=canonical(packet)
    catalog=json.loads((ROOT/'policy_catalogue.json').read_text())
    cells=catalog['cells']
    candidates=[]
    for assignment in product(ARMS,repeat=len(KEYS)):
        rows=[cells[k][a] for k,a in zip(KEYS,assignment)]
        metrics=assess(rows,request['train_shots_per_basis'],catalog['timings_us'])
        failed,unknown=check(assignment,metrics,request['constraints'])
        candidates.append(dict(assignment=dict(zip(KEYS,assignment)), metrics=metrics,
                               failed=failed,unknown=unknown))
    admissible=[c for c in candidates if not (c['failed'] or c['unknown'])]
    undecided=[c for c in candidates if not c['failed'] and c['unknown']]
    winner=min(admissible,key=lambda c:c['metrics']['expected_time_ms'],default=None)
    status='uncertified' if undecided else 'infeasible_in_catalogue'
    program=None
    if winner:
        ms=winner['metrics']['expected_time_ms']
        status='optimal_in_catalogue' if all(c['metrics']['expected_time_ms']>=ms for c in undecided) else 'best_certified'
        program=dict(bindings={k:cells[k][arm] for k,arm in winner['assignment'].items()},
                     decoder_rules={'verified':'Keep if verifier syndrome and verifier parity vanish. XOR data parity with decoder_lookup[data syndrome].',
                                    'terminal':'Keep if data syndrome vanishes. Return raw data parity.'},
                     controller={'training_shots_per_basis':request['train_shots_per_basis'],
                                 'validation_shots_per_basis':64,'feedback':'Evaluate 0,+1; stop on strict improvement. Otherwise evaluate -1 and compare against 0, retaining 0 on ties. Independently validate selection.'})
    return dict(status=status,request=request,selected=winner,candidates=candidates,program=program,
                comparison_class='All 2^6 assignments of two native instruments at six fixed angle/basis ports; one stationary assignment across every retry, training and validation.',
                platform={'noise':catalog['noise'],'timings_us':catalog['timings_us']},metric_scopes=SCOPES)


def native_readouts(row,arm,count,rng,noise):
    """Sample each actual native Clifford gate's independent Pauli fault."""
    n=row['peak_atoms']
    pauli=(rng.random((count,n))<noise['reset_x_error']).astype(np.uint8)
    for gate,transfer in zip(row['native_schedule']['gates'],row['pauli_transfers'],strict=True):
        qs=gate['qubits']
        packed=sum(pauli[:,q]<<(2*j) for j,q in enumerate(qs))
        image=np.asarray(transfer,dtype=np.uint8)[packed]
        d=1<<len(qs)
        F=noise['one_qubit_gate_average_fidelity' if d==2 else 'two_qubit_gate_average_fidelity']
        fault=rng.random(count)<((d+1)/d)*(1-F)
        image[fault]^=rng.integers(1,d*d,np.count_nonzero(fault),dtype=np.uint8)
        for j,q in enumerate(qs):
            pauli[:,q]=(image>>(2*j))&3
    ideal=row['ideal_pauli_expectation']
    parity=rng.integers(0,2,count,dtype=np.uint8) if ideal==0 else np.full(count,int(ideal<0),np.uint8)
    # Ideal basis outputs are uniform on the dual-Hamming coset of the sign.
    # The verifier is a separate logical zero block in the noiseless circuit.
    generators=np.array([[(q+1)>>j&1 for q in range(7)] for j in range(3)],np.uint8)
    blocks=[]
    for start in range(0,n,7):
        data=(rng.integers(0,2,(count,3),dtype=np.uint8)@generators)&1
        if start==0:
            data^=parity[:,None]
        blocks.append(data)
    bits=np.hstack(blocks)^(pauli&1)^(rng.random((count,n))<noise['readout_flip'])
    def syndrome(block):
        return sum(((block[:,cs].sum(axis=1).astype(np.uint8)&1)<<j) for j,cs in enumerate(CHECKS))
    data=bits[:,:7]
    check=syndrome(data)
    sign=data.sum(axis=1).astype(np.uint8)&1
    if arm=='verified':
        anc=bits[:,7:]
        keep=(syndrome(anc)==0)&((anc.sum(axis=1)&1)==0)
        sign^=np.asarray(row['decoder_lookup'],np.uint8)[check]
    else:
        keep=check==0
    return sign[keep],[''.join(str(int(b)) for b in obs) for obs in bits]


def replay_policy(plan,seed):
    if type(seed) is not int or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    if not isinstance(plan,dict) or plan!=select_policy(plan.get('request')):
        raise ValueError('The request, evidence and native program must match a freshly extracted plan.')
    if plan['selected'] is None:
        raise ValueError('This plan has no certified implementation to execute.')
    rng=np.random.default_rng(seed)
    gates=depth=attempts=accepted=0
    duration=0.
    batches=[]
    n=plan['request']['train_shots_per_basis']
    selected=plan['selected']['assignment']
    noise=plan['platform']['noise']
    clocks=plan['platform']['timings_us']
    def energy(t,m):
        nonlocal gates,depth,duration,attempts,accepted
        plus=[]
        records=[]
        for b in B:
            key=f'{t}/{b}'
            row=plan['program']['bindings'][key]
            missing=m
            ones=tries=0
            histogram={}
            while missing:
                sign,readouts=native_readouts(row,selected[key],missing,rng,noise)
                tries+=missing
                missing-=len(sign)
                ones+=int(sign.sum())
                for raw in readouts:
                    histogram[raw]=histogram.get(raw,0)+1
            plus.append(m-ones)
            attempts+=tries;accepted+=m
            gates+=tries*row['gate_count'];depth+=tries*row['depth']
            duration+=tries*(row['unitary_us']+clocks['fresh_reset_us_per_shot']+
                             clocks['parallel_image_us_per_shot']+clocks['decode_us_per_shot'])
            records.append(dict(key=key,arm=selected[key],attempts=tries,accepted=m,
                                plus=m-ones,raw_atom_bitstrings=histogram))
        duration+=clocks['controller_us_per_energy_batch']
        value=-(2*plus[0]-m)/m-.75*(2*plus[1]-m)/m
        batches.append(dict(binding=t,measurement=records,energy=value))
        return value
    e0=energy(0,n)
    if energy(1,n)<e0:
        t=1
    else:
        t=-1 if energy(-1,n)<e0 else 0
    fresh=energy(t,64)
    return dict(seed=seed,model='Native Clifford Pauli-frame replay of the independent gate/reset/readout model',
                selected_binding=t,fresh_energy=fresh,application_success=t==1 and abs(fresh+1)<=.2,
                gates=gates,depth=depth,time_ms=duration/1000,attempts=attempts,accepted=accepted,
                peak_atoms=max(plan['program']['bindings'][row['key']]['peak_atoms'] for b in batches for row in b['measurement']),history=batches)
