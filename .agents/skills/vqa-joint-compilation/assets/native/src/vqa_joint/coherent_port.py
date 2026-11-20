"""Process fidelity on the same coherent input and abstract output for every realization."""
from dataclasses import dataclass
import json
import numpy as np

from vqa_joint.host_policy import PolicyRefusal
from vqa_joint.sector_vqe.affine_port import AffinePort, recognize as recognize_affine
from vqa_joint.native_noise import evolve_density, local_unitary
from vqa_joint.sector_qaoa.parity_port import apply_binary, parity_ports
from vqa_joint.sector_qaoa.permutation_orbit import TRIANGLE_ORBITS, recognize_two_orbit


H = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
PREP_ORBIT = np.array([[.5, -np.sqrt(3)/2], [np.sqrt(3)/2, .5]])
CX = np.eye(4)[:, [0,3,2,1]]


def hadamards(n):
    result = np.ones((1,1))
    for _ in range(n):
        result = np.kron(H,result)
    return result


def orbit_embedding(groups, width):
    result = np.zeros((1<<width,len(groups)))
    for i, group in enumerate(groups):
        result[list(group),i] = 1 / np.sqrt(len(group))
    return result


def check_embedding(checks, width):
    groups = [[] for _ in range(1 << len(checks))]
    for x in range(1 << width):
        groups[apply_binary(checks,x)].append(x)
    return orbit_embedding(groups,width)


def apply_gate(columns, matrix, wires, width):
    local = [width-1-q for q in reversed(wires)]
    order = local + [a for a in range(width) if a not in local] + [width]
    shape = [2]*width+[columns.shape[1]]
    arranged = columns.reshape(shape).transpose(order)
    updated = matrix @ arranged.reshape(len(matrix),-1)
    return updated.reshape(arranged.shape).transpose(np.argsort(order)).reshape(columns.shape)


@dataclass(frozen=True)
class AffineSpace:
    entry: np.ndarray
    target: np.ndarray
    reduced: np.ndarray
    port: AffinePort


def source_target(ops, port):
    n = sum(op[0] == "measure" for op in ops)
    body = ops[:-n]
    if port == "affine_two_word":
        affine = recognize_affine(body,n)
        if affine is None or (1 << n)*2 > 512:
            raise ValueError("Binding does not implement the declared affine two-word port within the dense Choi limit")
        return AffineSpace(*affine.coherent_maps(), affine)
    if sorted((g,w,a) for g,w,a in body[:n]) != [("h",(i,),None) for i in range(n)]:
        raise ValueError("Coherent quotient ports require all-plus preparation")
    parity = dict(parity_ports(body[n:],n))
    if port == "triangle_two_orbits" and recognize_two_orbit(body,n) is not None:
        post_prep = orbit_embedding(TRIANGLE_ORBITS,n) @ PREP_ORBIT
    elif port == "binary_parity_first" and "parity_first" in parity:
        checks = parity["parity_first"].checks
        post_prep = check_embedding(checks,n) @ hadamards(len(checks))
    else:
        raise ValueError("Binding does not implement the declared coherent port")
    if (1<<n)*post_prep.shape[1] > 512:
        raise ValueError("Dense Choi evaluation is limited to 512 system-reference dimensions")
    source = hadamards(n) @ post_prep
    target = post_prep.astype(complex)
    for g,w,theta in body[n:]:
        if g == "cx":
            matrix = CX
        elif g == "rx":
            matrix = np.cos(theta/2)*np.eye(2)-1j*np.sin(theta/2)*np.array([[0,1],[1,0]])
        elif g in ("rz","rzz"):
            spectrum = [1,-1] if g == "rz" else [1,-1,-1,1]
            matrix = np.diag(np.exp(-.5j*theta*np.array(spectrum)))
        else:
            raise ValueError("Unsupported coherent task gate")
        target=apply_gate(target,matrix,w,n)
    return source, post_prep, target, parity


def fidelity(call, task_space, noise):
    c=call.readout
    if isinstance(task_space, AffineSpace):
        if c.kind == "identity" and c.native_width == len(task_space.port.axes):
            input_port, target_port = task_space.entry, task_space.target
        elif c.kind in ("affine_two_word", "fanout_X", "fanout_Z") and c.native_width == 1:
            a = task_space.port
            for b in (0,1):
                expected = sorted(apply_binary(a.rows,b | (r << 1)) ^ a.offset
                                  for r in range(1 << a.random_width))
                actual = sorted(c.decode_bits(b,r) for r in range(c.choice_count(b)))
                if actual != expected:
                    raise PolicyRefusal("Readout kernel disagrees with the affine task port")
            input_port, target_port = np.eye(2), task_space.reduced
        else:
            raise PolicyRefusal("Candidate does not implement the affine task port")
    else:
        source,post_prep,target,parity=task_space
        if c.kind=="identity":
            input_port, target_port = source, target
        elif c.kind in ("binary_parity","global_flip","orbit_sample"):
            groups=[[c.decode_bits(y,i) for i in range(c.choice_count(y))] for y in range(1<<c.native_width)]
            exit_map=orbit_embedding(groups,c.width)
            entry_map=(check_embedding(parity[call.variant].checks,c.width) if c.kind=="binary_parity" else exit_map)
            prep=PREP_ORBIT if c.kind=="orbit_sample" else hadamards(c.native_width)
            input_port=prep.T @ entry_map.T @ post_prep
            target_port=exit_map.T @ target
            if not np.allclose(exit_map @ target_port,target,rtol=0,atol=1e-10):
                raise PolicyRefusal("Required coherent output extends beyond this representation")
        else:
            raise PolicyRefusal("No common coherent embedding for this measured candidate")
    k=input_port.shape[1]
    if not np.allclose(input_port.conj().T @ input_port,np.eye(k),rtol=0,atol=1e-10):
        raise PolicyRefusal("Required coherent input extends beyond this representation")
    reorder=[call.lowered.wire_port.logical_outcome(x) for x in range(1<<c.native_width)]
    input_native,target_native=input_port[reorder],target_port[reorder]
    native=json.loads(call.lowered.schedule.read_text())
    if native["num_qubits"]!=c.native_width:
        raise ValueError("Choi input width disagrees with bound native schedule")
    ideal=input_native
    for gate in native["gates"]:
        ideal=apply_gate(ideal,local_unitary(gate),gate["qubits"],c.native_width)
    overlap=np.vdot(target_native,ideal)/k
    if abs(overlap) < 1e-12 or np.max(abs(ideal-target_native*overlap/abs(overlap)))>1e-8:
        raise PolicyRefusal("Emitted ideal channel fails the common task-port isometry")
    state=input_native.T.ravel()/np.sqrt(k)
    bell_target=target_native.T.ravel()/np.sqrt(k)
    rho=evolve_density(np.outer(state,state.conj()),c.native_width+k.bit_length()-1,native["gates"],noise)
    value=float(np.vdot(bell_target,rho @ bell_target).real)
    if not -1e-10 <= value <= 1+1e-10 or abs(np.trace(rho)-1)>1e-10:
        raise ArithmeticError("Invalid normalized Choi overlap")
    return min(1.,max(0.,value))
