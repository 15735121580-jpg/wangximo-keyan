"""Branch-and-bound on a platform adapter's complete physical implementation domain.

The adapter provides a mathematical coverage proof for its root and partitions.
Its lower bounds apply to every contract-feasible completion in a cell. The
verifier independently bounds the cost of an executable witness in the root,
including the delivery contract and any incumbent protections. Bounds use
integers or fractions in one registered unit. An infeasibility proof is
distinct from an unresolved cell.
"""
from fractions import Fraction
import heapq
from itertools import count


INFEASIBLE = object()


def _quantity(value):
    if type(value) not in (int, Fraction):
        raise ValueError("Provide a certified integer or rational cost bound")
    return Fraction(value)


def optimize_physical_domain(domain, verify_upper, *, initial=(), tolerance=0, max_expansions=None, prefer_exact=False, suggest=None):
    """Return a witness with an absolute gap, or the open interval at interruption.

    domain.root covers the declared problem, domain.lower(cell) returns a
    sound cost bound or INFEASIBLE, and domain.proposals(cell) supplies optional
    executables. domain.split(cell) returns a finite exhaustive cover, or None
    for an unresolved leaf. verify_upper(program) returns a certified upper
    cost or None. A feasible initial incumbent is optional. LLM proposals have
    no role in proving coverage or lower bounds.
    """
    epsilon = _quantity(tolerance)
    if epsilon < 0 or (max_expansions is not None and
                        (type(max_expansions) is not int or max_expansions < 0)):
        raise ValueError("Use a nonnegative tolerance and expansion budget")
    queue, unresolved = [], []
    serial = count()
    witness = upper = None
    expansions = 0

    def consider(proposals):
        nonlocal witness, upper
        for program in proposals:
            cost = verify_upper(program)
            if cost is None:
                continue
            cost = _quantity(cost)
            if upper is None or cost < upper:
                upper, witness = cost, program

    def add(cell, parent_lower=None):
        value = domain.lower(cell)
        if value is INFEASIBLE:
            return
        bound = _quantity(value)
        if parent_lower is not None:
            bound = max(parent_lower, bound)
        heapq.heappush(queue, (bound, next(serial), cell))

    consider(initial)
    add(domain.root)
    root_lower = queue[0][0] if queue else None
    while True:
        if upper is not None and (root_lower is None or root_lower > upper):
            raise ValueError("The root lower bound contradicts the verified witness")
        possible = []
        if queue:
            possible.append(queue[0][0])
        if unresolved:
            possible.append(unresolved[0])
        if upper is not None:
            possible.append(upper)
        lower = min(possible) if possible else None
        gap = upper - lower if upper is not None else None
        stopped = not queue or (max_expansions is not None and expansions >= max_expansions)
        if gap is not None and gap <= epsilon and (gap == 0 or not prefer_exact or stopped):
            status = "optimal" if gap == 0 else "epsilon_optimal"
            break
        if stopped:
            status = "infeasible" if lower is None else "open"
            break
        bound, _, cell = heapq.heappop(queue)
        if upper is not None and bound >= upper:
            continue
        expansions += 1
        if suggest is not None and expansions == 1:
            consider(suggest())
        if upper is None or bound < upper:
            consider(domain.proposals(cell))
        if upper is not None and root_lower > upper:
            raise ValueError("The root lower bound contradicts the verified witness")
        if upper is not None and bound >= upper:
            continue
        children = domain.split(cell)
        if children is None:
            heapq.heappush(unresolved, bound)
            continue
        children = tuple(children)
        if not children:
            raise ValueError("An empty split requires an explicit infeasibility proof")
        for child in children:
            add(child, bound)
    return dict(status=status, policy=witness, lower=lower, upper=upper, gap=gap,
                tolerance=epsilon, expansions=expansions)


def compile_joint(task, platform, *, tolerance=None):
    """Compile a task on a hardware adapter and return one executable plus bounds.

    The prepared problem owns the original host, its guarded cost verifier, and
    a covering domain. Its scope names the realizable controls or the restricted
    family covered by the proof. Domain proposals may use an LLM. The verifier
    must reject programs outside that domain, including changes to protected
    task semantics or host metrics. Bounds and tolerance use one objective unit.
    An explicit tolerance overrides the adapter default. Only a certified
    optimum or a witness within that tolerance occupies the delivery field.
    An open result keeps its feasible witness separately for search diagnostics.
    """
    problem = platform.prepare(task)
    if not isinstance(problem.scope, str) or not problem.scope.strip():
        raise ValueError("Name the implementation domain covered by the lower bounds")
    model_proposals = getattr(platform, "propose_programs", None)
    def suggestions():
        return model_proposals(task, problem.host)

    result = optimize_physical_domain(
        problem.domain, problem.verify_upper, initial=(problem.host,),
        tolerance=problem.tolerance if tolerance is None else tolerance,
        max_expansions=problem.max_expansions, prefer_exact=True,
        suggest=suggestions if model_proposals is not None else None,
    )
    if result["status"] == "open":
        result["incumbent"] = result["policy"]
        result["policy"] = None
    return dict(result, scope=problem.scope)
