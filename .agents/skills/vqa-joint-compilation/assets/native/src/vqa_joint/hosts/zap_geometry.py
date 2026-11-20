"""Geometry checks reused from the project ZAP audit (oq-original-tasks/zap/compare.py)."""
from collections import Counter, defaultdict
import math

def cells(arch,zone):
    return {(s['location'][0]+i*s['site_seperation'][0],s['location'][1]+j*s['site_seperation'][1])
           for z in arch[zone] for s in z['slms'] for i in range(s['c']) for j in range(s['r'])}

def motion_issues(instruction, held, positions):
    """Audit emitted paths, including parked or waiting atoms on this AOD."""
    issues = set()
    dest = {z['id']: (z['x_end'], z['y_end']) for z in instruction['locs']}
    moving = set(dest)
    on_aod = held | moving
    for axis in (0, 1):
        # One source line has one destination. Ordered distinct lines must
        # finish in strictly increasing order (author Router.compatible_2D).
        line_ends = {}
        for q in on_aod:
            start = positions[q][axis]
            end = dest.get(q, positions[q])[axis]
            if start in line_ends and line_ends[start] != end:
                issues.add('AOD_line_splits_or_crosses')
            line_ends[start] = end
        ends = [line_ends[start] for start in sorted(line_ends)]
        if any(left >= right for left, right in zip(ends, ends[1:])):
            issues.add('AOD_line_splits_or_crosses')
    stationary = {p for q, p in positions.items() if q not in moving}
    if len(set(dest.values())) != len(dest) or any(p in stationary for p in dest.values()):
        issues.add('occupied_motion_destination')
    # ZAP Router.verify_path tests lattice sites along each straight move.
    for q, end in dest.items():
        x, y = positions[q]
        dx, dy = end[0] - x, end[1] - y
        intervals = math.gcd(dx, dy)
        if intervals > 1 and any((x + i * (dx // intervals), y + i * (dy // intervals)) in stationary
                                 for i in range(1, intervals)):
            issues.add('straight_path_through_stationary_atom')
            break
    return sorted(issues)


def inspect(ir,arch):
    # Source order including each atom's own motions. A waiting atom in an
    # entanglement zone also has a dependency at every global CZ illumination.
    ent=cells(arch,'entanglement_zones');traps=ent|cells(arch,'storage_zones');edge=min(y for x,y in ent);pos={};held=set();counter=Counter()
    target=[];last=defaultdict(lambda:-1);slots=[];seen=defaultdict(list)
    first_bad=None
    def problem(k,t):
        nonlocal first_bad
        counter[k]+=1
        if first_bad is None:first_bad=(k,t)
    for t,e in enumerate(ir['instructions']):
        kind=e['type']
        if kind=='Init':
            pos={z['id']:(z['x'],z['y']) for z in e['locs']}
            if set(pos)!=set(range(ir['n_q'])) or len(set(pos.values()))!=len(pos):problem('initial_occupancy',t)
            if any(xy not in traps for xy in pos.values()):problem('initial_atom_off_SLM', t)
            continue
        if kind=='1qGate':
            slots.append(t)
            if len(e['qs'])!=len(set(e['qs'])):problem('same_atom_twice_in_1q_slot',t)
            for a,q in enumerate(e['qs']):
                z=e['locs'][a];point=z['x'],z['y']
                if pos[q]!=point:problem('emitted_1q_coordinate',t)
                target.append((t,a,q,last[q]));seen[q].append((t,a))
            continue
        if kind=='Crosstalk':
            for q in e['qs']:last[q]=t
            continue
        qs=e['qs']
        if kind in ('Park', 'BigMove'):
            for issue in motion_issues(e, held, pos):
                problem(issue, t)
        if kind == 'Activate':
            # Independent x and y beams select a Cartesian product; a
            # stationary SLM atom on any selected crosspoint is picked up too.
            select = set(qs)
            xs = {pos[q][0] for q in select}
            ys = {pos[q][1] for q in select}
            if any(q not in held and q not in select and x in xs and y in ys
                   for q, (x, y) in pos.items()):
                problem('unintended_AOD_pickup', t)
        if kind == 'Deactivate':
            select = set(qs)
            xs = {pos[q][0] for q in select}
            ys = {pos[q][1] for q in select}
            if any(pos[q][0] in xs and pos[q][1] in ys for q in held - select):
                problem('unintended_AOD_release', t)
            if any(pos[q] not in traps for q in select):
                problem('release_off_SLM', t)
        if kind=='2qGate':
            actual=set()
            occ={p:q for q,p in pos.items() if p[1]>=edge}
            if len(occ)!=sum(p[1]>=edge for p in pos.values()):problem('coincident_entanglement_sites',t)
            # Pulses are delivered after all handovers. The entangling region is
            # ZAP Router.write_idle's y>=edge. At valid trap coordinates only
            # same-row 4-um pairs lie at positive distance <=5 um.
            on_grid=all(p in ent for p in occ)
            if not on_grid:problem('operation_off_entangling_trap_grid',t)
            for (x,y),q in occ.items():
                if on_grid:
                    r=occ.get((x+4,y))
                    if r is not None:actual.add(tuple(sorted((q,r))))
                else:
                    for (u,v),r in occ.items():
                        if q<r and (x-u)**2+(y-v)**2<=25:actual.add((q,r))
                last[q]=t
            wanted={tuple(sorted(z)) for z in e['gates']}
            if actual!=wanted:problem('global_CZ_partner_set',t)
            expected_idle={q for q,(x,y) in pos.items() if y>=edge}-set(qs)
            next_ev=ir['instructions'][t+1:t+2]
            emitted_idle=set(next_ev[0]['qs']) if next_ev and next_ev[0]['type']=='Crosstalk' else set()
            if emitted_idle!=expected_idle:problem('spectator_list',t)
            if len(qs)!=len(set(qs)):problem('multiple_CZ_on_one_qubit',t)
        for z in e.get('locs',[]):
            q=z['id'];last[q]=t
            if 'x_begin' in z:
                if q not in held:problem('MOVE_before_activation',t)
                if pos[q]!=(z['x_begin'],z['y_begin']):problem('move_start_mismatch',t)
                pos[q]=(z['x_end'],z['y_end'])
            else:
                if pos[q]!=(z['x'],z['y']):problem('gate_or_handover_site_mismatch',t)
                if kind=='Activate':
                    if q in held:problem('repeated_Activate_without_Deactivate',t)
                    held.add(q)
                elif kind=='Deactivate':
                    if q not in held:problem('Deactivate_without_Activate',t)
                    held.discard(q)
    if any(len(v)!=len(set(v)) for v in seen.values()):raise AssertionError('duplicate pulse identities')
    return slots,target,counter,first_bad

