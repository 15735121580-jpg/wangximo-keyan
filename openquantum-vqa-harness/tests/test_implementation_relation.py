import unittest
from dataclasses import replace
from itertools import product

from vqa_harness import (Assessment, HostJob, OpenDiagram, Port, ReleaseContract,
                         Stage, direct_diagram, implementation_relation,
                         replace_region, search_diagram)


def job_for(stages, roles):
    graph = OpenDiagram(stages[0].inputs, tuple(stages), stages[-1].outputs)
    return HostJob("preserve returned logical bit", "test device", graph, roles,
                   ReleaseContract("wire-label model", ("cost",)))


def stage(name, role, before, after, key="ordinary"):
    return Stage(name, role, (before,), (after,), key)


class LogicalBitVerifier:
    model = "wire-label model"

    def assess(self, contract, graph):
        # Reachable outcomes are both binary values. Physical work is counted
        # over the complete job; the alternative preparation is more expensive.
        for incoming in (0, 1):
            bit = incoming
            for s in graph.stages:
                if s.implementation in ("encode complemented", "invert score"):
                    bit ^= 1
            if bit != incoming:
                return Assessment("infeasible")
        costs = {"encode complemented": 4, "read in complemented frame": 0, "invert score": 0}
        return Assessment("feasible", sum(costs.get(s.implementation, 2) for s in graph.stages))

    def upper_bounds(self, contract, before, after):
        return {"cost": after.evidence - before.evidence}

    upper_bounds_against = upper_bounds


class ImplementationRelationTests(unittest.TestCase):
    def setUp(self):
        enter, u, v, w, result = [Port(name, "bit") for name in ("entry", "u", "v", "w", "return")]
        self.stages = [stage("a", "prepare", enter, u), stage("b", "detect", u, v),
                       stage("c", "decode", v, w), stage("d", "controller", w, result)]
        self.job = job_for(self.stages, ("prepare", "detect", "decode"))
        uc, vc = (replace(port, space="complemented bit") for port in (u, v))
        self.variants = dict(a=stage("a", "prepare", enter, uc, "encode complemented"),
                             b=stage("b", "detect", uc, vc, "read in complemented frame"),
                             c=stage("c", "decode", vc, w, "invert score"))

    def test_atomic_three_stage_rewrite_through_both_public_algorithms(self):
        for i, s in enumerate(self.variants.values()):
            original = self.stages[i]
            with self.assertRaisesRegex(ValueError, "typed incoming and outgoing ports"):
                replace_region(self.job.host, (s.id,), OpenDiagram(s.inputs, (s,), s.outputs),
                               self.job.stages)
            self.assertEqual(original.id, s.id)
        def options(job, graph, s):
            self.assertIn(s.id, self.variants)
            return (self.variants[s.id], self.variants[s.id])

        relation = implementation_relation(options)
        matches = tuple(relation(self.job, self.job.host))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].selected, ("a", "b", "c"))
        rules = (("coordinate physical frame and score", relation),)
        verifier = LogicalBitVerifier()
        a1 = direct_diagram(self.job, rules, verifier, max_rewrites=1)
        a2 = search_diagram(self.job, rules, verifier, max_rewrites=1, max_candidates=3,
                            incumbent=a1)
        self.assertEqual(a1["status"], "released")
        self.assertEqual(a2["status"], "retained_incumbent")
        self.assertEqual(a2["policy"], a1["policy"])
        self.assertEqual(a2["policy"].stages[-1], self.stages[-1])
        self.assertEqual(verifier.assess(self.job.contract, a2["policy"]).evidence, 6)
        self.assertEqual(verifier.assess(self.job.contract, self.job.host).evidence, 8)

    def test_external_and_fixed_boundaries_hold_even_if_an_adapter_changes_them(self):
        first, _, last, _ = self.stages
        bad_input = replace(first, inputs=(replace(first.inputs[0], space="changed"),))
        before_fixed = replace(last, outputs=(replace(last.outputs[0], space="changed"),))
        opts = {first.id: (bad_input,), last.id: (before_fixed,)}
        matcher = implementation_relation(lambda j,g,s: opts.get(s.id, ()))
        direct = direct_diagram(self.job, (("try", matcher),), LogicalBitVerifier(), 1)
        self.assertEqual(direct["status"], "retained_host")
        self.assertEqual(direct["branches"], [])
        self.assertIs(direct["policy"], self.job.host)

    def test_sparse_dependency_graph_matches_cartesian_oracle(self):
        a, b, x, y, z, end = [Port(n, "native") for n in ("a","b","x","y","z","end")]
        split = Stage("split", "r", (), (a,b))
        left = stage("left", "r", a, x)
        right = stage("right", "r", b, y)
        join = Stage("join", "r", (x,y), (z,))
        decode = stage("decode", "r", z, end)
        job = job_for((split,left,right,join,decode), ("r",))
        variants = {}
        for s in job.host.stages:
            exposed = len(s.inputs)+len(s.outputs)
            opts = []
            for spaces in product(("native","mapped"), repeat=exposed):
                ports = tuple(replace(p,space=v) for p,v in zip((*s.inputs,*s.outputs),spaces))
                opts.append(replace(s, inputs=ports[:len(s.inputs)], outputs=ports[len(s.inputs):]))
            variants[s.id] = tuple(opts)
        # Each of the five internal edges may independently take one of two
        # types. The outermost output stays native. Exclude only the original.
        matcher = implementation_relation(lambda j,g,s: variants[s.id])
        observed = set()
        for pick in matcher(job, job.host):
            result = replace_region(job.host, pick.selected, pick.replacement, job.stages)
            observed.add(tuple(next(p.space for s in result.stages for p in s.outputs if p.name == n)
                               for n in ("a","b","x","y","z")))
        expected = set(product(("native","mapped"), repeat=5)) - {("native",)*5}
        self.assertEqual(observed, expected)

    def test_long_fixed_chain_and_invalid_catalog(self):
        ports = [Port(str(i), "signal") for i in range(1300)]
        stages = [stage(str(i), "r", lo, hi) for i,(lo,hi) in enumerate(zip(ports, ports[1:]))]
        job = job_for(stages, ("r",))
        matcher = implementation_relation(
            lambda j,g,s: (replace(s, implementation="other"),) if s == stages[-1] else ())
        self.assertEqual([m.selected for m in matcher(job, job.host)], [(stages[-1].id,)])
        malformed = replace(stages[-1], inputs=(Port("renamed", "signal"),))
        reject = implementation_relation(lambda j,g,s: (malformed,) if s == stages[-1] else ())
        with self.assertRaisesRegex(ValueError, "named interfaces"):
            list(reject(job, job.host))


if __name__ == "__main__":
    unittest.main()
