"""Pull local terminal measurements through an affine two-word isometry."""
from dataclasses import dataclass
from math import isfinite

import numpy as np


I = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], complex)
Y = np.array([[0,-1j],[1j,0]])
Z = np.diag([1.,-1.])
H = (X+Z)/np.sqrt(2)
S = np.diag([1.,1j])
T = np.diag([1.,np.exp(1j*np.pi/4)])
FIXED = {'h':H,'x':X,'y':Y,'z':Z,'s':S,'sdg':S.conj(),'t':T,'tdg':T.conj()}
PAULI = {'rx':X,'ry':Y,'rz':Z}
BASIS = {'Z':I,'X':H,'Y':H @ S.conj()}


@dataclass(frozen=True)
class AffinePort:
    active: int
    prefix: tuple
    words: tuple
    axes: tuple
    logical_basis: str
    rows: tuple
    offset: int
    random_width: int

    @property
    def operations(self):
        ops = [(g,(0,),a) for g,_,a in self.prefix]
        if self.logical_basis == 'Y':
            ops.append(('sdg',(0,),None))
        if self.logical_basis != 'Z':
            ops.append(('h',(0,),None))
        return tuple(ops+[('measure',(0,),None)])

    def coherent_maps(self):
        """Return the input embedding K, target B V U and reduced target D U."""
        prep = I
        for g,_,angle in self.prefix:
            op = FIXED[g] if g in FIXED else (np.cos(angle/2)*I-1j*np.sin(angle/2)*PAULI[g])
            prep = op @ prep
        columns = []
        for word in self.words:
            v = np.ones(1,complex)
            for q in reversed(range(len(self.axes))):
                v = np.kron(v, BASIS[self.axes[q]][:, (word >> q) & 1])
            columns.append(v)
        k = np.zeros((1<<len(self.axes),2),complex)
        k[0,0] = k[1<<self.active,1] = 1.
        return k, np.column_stack(columns) @ prep, BASIS[self.logical_basis] @ prep


def readout(words, axes):
    """Encode the conditional readout as y = rows·(b,r) XOR offset over GF(2)."""
    w0,w1 = words
    diff = w0 ^ w1
    trans = [q for q,a in enumerate(axes) if a != 'Z']
    zmask = sum(1 << q for q,a in enumerate(axes) if a == 'Z')
    rows = [0]*len(axes)
    offset = w0 & zmask
    if diff & zmask:
        for q in range(len(axes)):
            if diff & zmask & (1 << q):
                rows[q] = 1
        for i,q in enumerate(trans,1):
            rows[q] = 1 << i
        return 'Z', tuple(rows), offset, len(trans)

    # The two words overlap in the measured Z positions. An X/Y result has
    # relative amplitude (-1)^<y,diff> * i^phase between the two input labels.
    ymask = sum(1 << q for q,a in enumerate(axes) if a == 'Y')
    phase = ((w0 & ymask).bit_count() - (w1 & ymask).bit_count()) % 4
    pivot = (diff & -diff).bit_length()-1
    rows[pivot] = 1
    for i,q in enumerate((q for q in trans if q != pivot),1):
        rows[q] = 1 << i
        if diff & (1 << q):
            rows[pivot] ^= rows[q]
    if phase in (1,2):
        offset ^= 1 << pivot
    return 'Y' if phase % 2 else 'X', tuple(rows), offset, len(trans)-1


def recognize(gates, width):
    """Read one active-wire prefix, an X/CX encoder and local terminal Z/X/Y bases."""
    if width < 2 or not gates or len(gates[0][1]) != 1:
        return None
    active, = gates[0][1]
    i = 0
    while i < len(gates):
        g,w,a = gates[i]
        valid = (g in FIXED and a is None) or (g in PAULI and a is not None and isfinite(a))
        if not valid or w != (active,):
            break
        i += 1
    if not i:
        return None
    prefix = tuple(gates[:i])
    words = [0, 1 << active]
    while i < len(gates) and gates[i][0] in ('x','cx'):
        g,w,a = gates[i]
        if a is not None or len(w) != (2 if g == 'cx' else 1):
            return None
        for j,word in enumerate(words):
            if g == 'x' or word & (1 << w[0]):
                words[j] ^= 1 << w[-1]
        i += 1
    bases = [[] for _ in range(width)]
    for g,w,a in gates[i:]:
        if g not in ('sdg','h') or len(w)!=1 or a is not None:
            return None
        bases[w[0]].append(g)
    lookup = {():'Z',('h',):'X',('sdg','h'):'Y'}
    if any(tuple(ops) not in lookup for ops in bases):
        return None
    axes = tuple(lookup[tuple(ops)] for ops in bases)
    return AffinePort(active,prefix,tuple(words),axes,*readout(words,axes))
