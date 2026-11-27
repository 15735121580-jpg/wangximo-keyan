"""Constructed four-stage, two-input/two-output example; no physical QEC claim."""
from fractions import Fraction
import itertools
import unittest
from vqa_harness.diagram_context import OpenDiagram, Port, Stage, replace_region
from vqa_harness.guarded_release import Assessment, ReleaseContract
from vqa_harness.host_compilation import Branch, HostJob, compile_bounded, compile_direct


ROLES = ("encoding","placement","routing","measurement")
q = Port("prepared","computational-basis bit")
mask = Port("frame","classical bit")
bit = Port("measured","classical measurement bit")
carried_mask = Port("reported_frame","classical bit")
inputs = (Port("initial_bit","computational-basis bit"),Port("initial_frame","classical bit"))
answer = Port("answer","scored pair")


def region(prefix):
    q1, f1 = Port(prefix+"-code","encoded basis bit"),Port(prefix+"-frame","classical bit")
    q2, f2 = Port(prefix+"-site","basis bit on virtual site"),Port(prefix+"-siteflag","classical bit")
    q3, f3 = Port(prefix+"-device","basis bit on device site"),Port(prefix+"-deviceflag","classical bit")
    stages = (Stage(prefix+"-encode","encoding",(q,mask),(q1,f1)),
              Stage(prefix+"-place","placement",(q1,f1),(q2,f2)),
              Stage(prefix+"-route","routing",(q2,f2),(q3,f3)),
              Stage(prefix+"-measure","measurement",(q3,f3),(bit,carried_mask)))
    return OpenDiagram((q,mask),stages,(bit,carried_mask))


REFERENCE = region("original")
CANDIDATE = region("candidate")
PREP = Stage("prepare","preparation",inputs,(q,mask))
SCORE = Stage("consume","controller",(bit,carried_mask),(answer,))
HOST = OpenDiagram(inputs,(PREP,*REFERENCE.stages,SCORE),(answer,))


def trace(diagram, value, frame):
    """Check every one of four task inputs and count the charged conditional bit flip."""
    state = {inputs[0]:value,inputs[1]:frame}
    gate_count = 0
    for stage in diagram.stages:
        v,k = (state[p] for p in stage.inputs)
        if stage.id == "original-encode":
            v ^= k
            gate_count += k
        elif stage.id == "original-measure":
            v ^= k   # Classical readout compensation
        elif stage.id == "consume":
            state[answer] = (v,k)
            continue
        state.update(zip(stage.outputs,(v,k)))
    return state[answer],gate_count


class Adapter:
    model = "four-stage-constructed-basis-and-frame"

    def verifier(self, job):
        return self

    def construct(self, job, branch):
        new = REFERENCE if branch.name == "identity" else CANDIDATE
        return replace_region(job.host,(s.id for s in REFERENCE.stages),new,branch.stages)

    def assess(self, contract, policy):
        evaluations = [trace(policy,v,k) for v,k in itertools.product(range(2),repeat=2)]
        outputs,counts = zip(*evaluations)
        if outputs != tuple(itertools.product(range(2),repeat=2)):
            return Assessment("infeasible","different classical output")
        return Assessment("feasible",Fraction(sum(counts),len(counts)))

    def upper_bounds(self, contract, original, candidate):
        return {"mean_controlled_flips":candidate.evidence-original.evidence,"output_loss":0}


class MultiStageTests(unittest.TestCase):
    def test_both_entry_points_replace_all_four_stages_together(self):
        job = HostJob("all four basis/frame pairs","constructed one-site basis model",HOST,ROLES,
                      ReleaseContract(Adapter.model,("mean_controlled_flips","output_loss")))
        proposal = Branch("keep-frame-classical",ROLES)
        direct = compile_direct(job,proposal,Adapter())
        searched = compile_bounded(job,(Branch("identity",ROLES),proposal),2,Adapter(),legacy_host_only=True)
        for r in (direct,searched):
            self.assertEqual(r["status"],"released")
            self.assertEqual(r["branches"][r["selected"]]["upper_differences"]["mean_controlled_flips"],Fraction(-1,2))
            self.assertEqual(len(r["branches"][r["selected"]]["stages"]),4)
            self.assertIs(r["policy"].stages[0],PREP)
            self.assertIs(r["policy"].stages[-1],SCORE)
        self.assertEqual(searched["selected"],1)

    def test_missing_stage_permission_and_mistyped_boundary(self):
        selected = [s.id for s in REFERENCE.stages]
        with self.assertRaises(ValueError):
            replace_region(HOST,selected,CANDIDATE,ROLES[:-1])
        wrong = Port("reported_frame","quantum register")
        altered = Stage("bad","measurement",(q,mask),(bit,wrong))
        with self.assertRaises(ValueError):
            replace_region(HOST,selected,OpenDiagram((q,mask),(altered,),(bit,wrong)),ROLES)

    def test_cycle_and_accidental_fanout(self):
        with self.assertRaises(ValueError):
            OpenDiagram((),(Stage("a","mapping",(q,),(mask,)),Stage("b","routing",(mask,),(q,))),())
        with self.assertRaises(ValueError):
            OpenDiagram((q,),(Stage("a","readout",(q,),(bit,)),Stage("b","readout",(q,),(carried_mask,))),
                        (bit,carried_mask))
