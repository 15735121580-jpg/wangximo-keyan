"""Execute a bounded-score training batch against an earlier, exact incumbent.

The largest sum wins; ties stay with the earlier candidate. The executor must
deliver fresh serial blocks on demand. A fixed independent terminal batch is
the caller's responsibility. A losing mean can be skipped only when no later
stage consumes it; adaptive callers normally qualify only on the last round.
"""
from numbers import Integral


def select_candidate(plan, index, incumbent, take):
    cap, count = plan["training_cap"], plan["training_candidates"]
    block, lo, hi = plan["block"], plan["score_lower"], plan["score_upper"]
    if any(type(value) is not int for value in (cap, count, block, lo, hi, index)):
        raise ValueError("Bounds, counts and candidate index must be integers")
    if min(cap, count, block) <= 0 or not 0 <= index < count or lo > hi:
        raise ValueError("Invalid batch or score bounds")
    if incumbent is not None and (type(incumbent) is not int or not lo * cap <= incumbent <= hi * cap):
        raise ValueError("The incumbent must be an exact sum from a complete training batch")
    if plan["skip_scope"] not in ("last", "all_after_first"):
        raise ValueError("Unknown scope for skipping a training mean")
    final = index == count - 1
    can_skip = incumbent is not None and (final or plan["skip_scope"] == "all_after_first")
    used, total = 0, 0
    while used < cap:
        size = min(block, cap - used)
        values = list(take(size))
        if len(values) != size or any(not isinstance(v, Integral) or isinstance(v, bool)
                                      or not lo <= v <= hi for v in values):
            raise ValueError("Executor returned an invalid block")
        total += sum(map(int, values))
        used += size
        remaining = cap - used
        if can_skip and total + remaining * hi <= incumbent:
            return False, None, used
        if can_skip and final and total + remaining * lo > incumbent:
            return True, None, used
    return incumbent is None or total > incumbent, total, used
