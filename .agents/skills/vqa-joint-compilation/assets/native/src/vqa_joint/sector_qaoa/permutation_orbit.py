"""Lower a symmetric triangle QAOA word on its two-dimensional invariant space."""
from math import atan2, cos, isfinite, sin, sqrt
from numbers import Real

import numpy as np


TRIANGLE_ORBITS = ((0, 7), (1, 2, 3, 4, 5, 6))
EDGES = {(0, 1), (0, 2), (1, 2)}
WIRES = {(0,), (1,), (2,)}
IDENTITY = np.eye(2)
MIXER_TRACELESS = np.array([[-1., sqrt(3.)], [sqrt(3.), 1.]])


def _uniform_angle(operations, name, required_ports):
    ports = set()
    angles = []
    for op, wires, angle in operations:
        if op != name or isinstance(angle, bool) or not isinstance(angle, Real) or not isfinite(angle):
            return None
        ports.add(tuple(sorted(wires)))
        angles.append(angle)
    if ports != required_ports or any(a != angles[0] for a in angles):
        return None
    return float(angles[0])


def _zyz_operations(unitary):
    # The native gate sequence Rz(lambda), Ry(theta), Rz(phi) realizes U up
    # to a global phase. Singular cases fix only the defined sum or difference.
    a, b, c, d = unitary.ravel()
    theta = 2 * atan2(abs(c), abs(a))
    if abs(c) < np.finfo(float).eps:
        phi, lam = 0., np.angle(d) - np.angle(a)
    elif abs(a) < np.finfo(float).eps:
        phi, lam = np.angle(c) - np.angle(-b), 0.
    else:
        phi, lam = np.angle(c) - np.angle(a), np.angle(-b) - np.angle(a)
    return (("rz", (0,), float(lam)), ("ry", (0,), theta),
            ("rz", (0,), float(phi)), ("measure", (0,), None))


def recognize_two_orbit(body, width):
    """Return (reduced measured operations, label classes) or None.

    body omits terminal measurements. The caller checks that the original
    terminal instrument is computational-basis readout on all three atoms.
    Each bound layer contains a common Rzz angle on the three edges and
    a common Rx angle on the three vertices. Unobservable global phases are
    omitted. Exact equality of bound angles is a symmetry guard.

    Let V embed |0>, |1> as the normalized orbit sums. With P=Ry(2pi/3)
    and E=H^{tensor 3} V P, the complete bound word U obeys U E = V u up
    to global phase. E|0> is the fresh three-atom |000> input.
    """
    if width != 3 or len(body) < 9 or (len(body) - 3) % 6:
        return None
    prep = body[:3]
    if any(op != "h" or angle is not None for op, _, angle in prep):
        return None
    if {tuple(wires) for _, wires, _ in prep} != WIRES:
        return None
    # The plus state has amplitudes 1/2 and sqrt(3)/2 on the two orbits.
    unitary = np.array([[0.5, -sqrt(3.) / 2], [sqrt(3.) / 2, 0.5]], dtype=complex)
    for start in range(3, len(body), 6):
        zz_angle = _uniform_angle(body[start:start + 3], "rzz", EDGES)
        x_angle = _uniform_angle(body[start + 3:start + 6], "rx", WIRES)
        if zz_angle is None or x_angle is None:
            return None
        # Eigenvalues of sum ZZ are (3, -1). After dropping a common phase
        # the relative cost phase is exp(2i*zz_angle). The restricted sum X
        # is I+M where M^2=4 I.
        phase = np.exp(1j * zz_angle) ** 2
        cost = np.diag([1., phase])
        mix = cos(x_angle) * IDENTITY - 0.5j * sin(x_angle) * MIXER_TRACELESS
        unitary = mix @ cost @ unitary
    return _zyz_operations(unitary), TRIANGLE_ORBITS
