import unittest
from fractions import Fraction as F

from policy_ir import Action, Decision, Leaf, Outcome, PublicPort
from prefix_compiler import (BranchingBound, InvalidGrowthCertificate, InvalidIncumbent,
                             Law, compile_with_incumbent, replay_time)


class Table:
    """Fixture transition rows: (action, conditional mean ticks, [(label,target,prob)])."""
    def __init__(self, rows):
        self.rows = rows

    def is_terminal(self, port, history):
        return port.type_name in {"delivered", "failure"}

    def actions(self, port, history):
        return tuple(Action(name, port.type_name, tuple(Outcome(label, PublicPort(target))
                                for label, target, p in branches))
                     for name, cost, branches in self.rows.get(port.type_name, ()))

    def law(self, port, history, action):
        for name, cost, branches in self.rows[port.type_name]:
            if name == action.name:
                return Law(cost, tuple(p for label, target, p in branches))
        raise AssertionError("unexpected action")


def deterministic_step(name, cost, target):
    return name, cost, (("next",target,1),)


def first_reference(table, root):
    def visit(s,h):
        if table.is_terminal(s,h):
            return Leaf(s,h,"delivered")
        a = table.actions(s,h)[0]
        return Decision(s,h,a,tuple((o,visit(o.target,h+((a.name,o.label),))) for o in a.outcomes))
    return visit(root,())


def accepts(_):
    return True


class PrefixTests(unittest.TestCase):
    def test_locally_slower_and_certified_expensive_cut(self):
        tab = Table({"s":(deterministic_step("fast",1,"inconvenient"),
                          deterministic_step("slower",3,"ready"),
                          deterministic_step("expensive",12,"delivered")),
                     "inconvenient":(deterministic_step("return",10,"delivered"),),
                     "ready":(deterministic_step("read",1,"delivered"),)})
        root = PublicPort("s")
        res = compile_with_incumbent(tab,root,first_reference(tab,root),accepts,
                                     BranchingBound(0,1,1))
        self.assertEqual(res.time,4)
        self.assertEqual(res.program.action.name,"slower")
        self.assertEqual(res.incumbent_time,11)
        self.assertEqual(res.depth_bound_from_contract,11)
        self.assertGreater(res.pruned_actions,0)

    def test_accept_rare_expensive_branch_and_never_drop_dead_one(self):
        tab = Table({"s":(deterministic_step("baseline",5,"delivered"),
                          ("short-mostly",1,(("+","delivered",F(9,10)),("-","slow",F(1,10)))),
                          ("incomplete",1,(("+","delivered",F(99,100)),("-","stuck",F(1,100))))),
                     "slow":(deterministic_step("costly-recovery",30,"delivered"),)})
        root = PublicPort("s")
        res = compile_with_incumbent(tab,root,first_reference(tab,root),accepts,
                                     BranchingBound(1,F(1,100),1))
        self.assertEqual(res.time,4)  # root cost 1, plus (1/10)*30, not 31
        self.assertEqual(res.program.action.name,"short-mostly")
        self.assertEqual(len(res.program.branches),2)
        self.assertEqual(res.dead_prefixes,1)

    def test_unbounded_deterministic_repeat_no_silent_timeout(self):
        tab = Table({"s":(deterministic_step("finish",5,"delivered"),
                          deterministic_step("repeat",1,"s"))})
        root = PublicPort("s")
        ref = first_reference(tab,root)
        res = compile_with_incumbent(tab,root,ref,accepts,BranchingBound(0,1,1))
        self.assertEqual(res.time,5)
        self.assertEqual(res.root_candidates,1)
        self.assertGreater(res.pruned_actions,1)
        with self.assertRaises(InvalidGrowthCertificate):
            compile_with_incumbent(tab,root,ref,accepts,BranchingBound(0,1,2))

    def test_infeasible_or_missing_incumbent_and_false_certificate(self):
        tab = Table({"s":(("draw",1,(("given","delivered",F(1,2)),
                                     ("rejected","failure",F(1,2)))),)})
        root = PublicPort("s")
        ref = first_reference(tab,root)
        self.assertEqual(replay_time(tab,root,ref),1)
        with self.assertRaises(InvalidIncumbent):
            compile_with_incumbent(tab,root,None,accepts)
        with self.assertRaises(InvalidIncumbent):
            compile_with_incumbent(tab,root,ref,lambda _:False)
        with self.assertRaises(InvalidIncumbent):
            replay_time(tab,root,Decision(root,(),ref.action,ref.branches[:1]))
        with self.assertRaises(InvalidGrowthCertificate):
            compile_with_incumbent(tab,root,ref,accepts,BranchingBound(0,F(1,2),1))
        res = compile_with_incumbent(tab,root,ref,accepts,BranchingBound(1,F(1,2),1))
        self.assertEqual(res.program.branches[1][1].port.type_name,"failure")

if __name__ == "__main__":
    unittest.main()
