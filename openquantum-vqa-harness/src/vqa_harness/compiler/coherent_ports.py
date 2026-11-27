"""Single-pass simplification for an arbitrary coherent input/output contract."""
from copy import deepcopy
from math import pi, atan2
from cmath import phase
from .native_ports import I, Z, EPS, multiply, monomial, project_frame, euler, local_matrix, metrics


def scalar(u):
    return max(abs(u[1]),abs(u[2]),abs(u[0]-u[3]))<EPS


def angles(u):
    kind=monomial(u)
    if kind==0:
        return [0.,0.,phase(u[3]/u[0])]
    if kind==1:
        return [pi,phase(u[2]/(-u[1])),0.]
    return euler(u)


def h_or_z(g):
    if g['gate']=='cz':
        return None
    u=local_matrix(g)
    if max(abs(u[0]-u[1]),abs(u[0]-u[2]),abs(u[0]+u[3]))<EPS:
        return 'H'
    if monomial(u)==0:
        return 'Z'
    return None


def row_rotation_fusion(gates):
    # One whole addressed H tile, a whole Z-rotation tile on the same columns,
    # and a second whole H tile become a single allowed X-rotation tile.
    tiles=[]
    for i,g in enumerate(gates):
        kind=h_or_z(g)
        if (not kind or not tiles or kind!=tiles[-1][0] or
                g['qubits'][0] in tiles[-1][1] or
                (kind=='Z' and local_matrix(g)!=local_matrix(gates[tiles[-1][2][0]]))):
            tiles.append((kind,set(),[]))
        tiles[-1][1].update(g['qubits'])
        tiles[-1][2].append(i)
    replacements={}
    drop=set()
    j=0
    while j+2<len(tiles):
        a,b,c=tiles[j:j+3]
        if (a[0],b[0],c[0])==('H','Z','H') and a[1]==b[1]==c[1]:
            u=local_matrix(gates[b[2][0]])
            theta=phase(u[3]/u[0])
            drop.update(a[2]);drop.update(b[2])
            for i in c[2]:
                replacements[i]={**deepcopy(gates[i]),'gate':'u3','params':[theta,-pi/2,pi/2]}
            j+=3
        else:
            j+=1
    return [(i,replacements.get(i,g)) for i,g in enumerate(gates) if i not in drop]


def coherent_frames(gates,n):
    frames=[I]*n
    kept=[]
    for i,g in enumerate(gates):
        if g['gate']=='cz':
            a,b=g['qubits']
            fa,fb=monomial(frames[a]),monomial(frames[b])
            if fa:
                frames[b]=multiply(Z,frames[b])
            if fb:
                frames[a]=multiply(Z,frames[a])
        else:
            q,=g['qubits']
            u=local_matrix(g)
            flip=monomial(u)
            if flip is not None:
                frames[q]=multiply(project_frame(u,flip),frames[q])
                continue
            if not scalar(frames[q]):
                g={**deepcopy(g),'gate':'u3','params':euler(multiply(u,frames[q]))}
            frames[q]=I
        kept.append((i,g))
    # This check only examines a product of symbolic port frames. A nontrivial
    # open endpoint must remain physical on a coherent job.
    if any(not scalar(f) for f in frames):
        return list(enumerate(gates))
    return kept


def fuse_wire_runs(pairs,n):
    pending=[[] for _ in range(n)]
    replacements={}
    drop=set()
    def flush(q):
        seq=pending[q]
        if len(seq)>1:
            u=I
            for i,g in seq:
                u=multiply(local_matrix(g),u)
                drop.add(i)
            if not scalar(u):
                i,g=seq[-1]
                replacements[i]={**deepcopy(g),'gate':'u3','params':angles(u)}
                drop.remove(i)
        pending[q]=[]
    for i,g in pairs:
        if g['gate']=='cz':
            for q in g['qubits']:
                flush(q)
        else:
            q,=g['qubits']
            pending[q].append((i,g))
    for q in range(n):
        flush(q)
    return [(i,replacements.get(i,g)) for i,g in pairs if i not in drop]


def rewrite_coherent_ports(source, *, local_control):
    """local_control is 'addressed-row-hrzhrx' or 'individual-u3'. Whole-channel contract."""
    n=source['num_qubits']
    gates=source['gates']
    for g in gates:
        q=g['qubits']
        if len(set(q))!=len(q) or any(not isinstance(i,int) or i not in range(n) for i in q):
            raise ValueError('Invalid native ports')
        if g['gate']=='cz':
            if len(q)!=2 or g.get('params',[]):
                raise ValueError('Invalid CZ')
        elif len(q)==1:
            local_matrix(g)
        else:
            raise ValueError('Unsupported native gate')
    if local_control=='individual-u3':
        pairs=fuse_wire_runs(coherent_frames(gates,n),n)
    elif local_control=='addressed-row-hrzhrx':
        pairs=row_rotation_fusion(gates)
    else:
        raise ValueError('Unsupported hardware control capability')
    kept=[i for i,_ in pairs]
    native={**deepcopy(source),'gates':[deepcopy(g) for _,g in pairs]}
    return dict(native_program=native,bit_xor_mask=0,source_slots=kept,
        omitted_slots={'zero_controlled_cz':[], 'monomial_local':[], 'coherent_local':sorted(set(range(len(gates)))-set(kept))},
        metrics=metrics(native['gates'],n),preconditions={'contract':'arbitrary coherent CPTP input and output; modulo scalar phase',
            'local_control':local_control,'calibrated_noise_model':'independent isotropic local depolarization'})
