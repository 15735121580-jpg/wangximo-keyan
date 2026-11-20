"""Executable short-circuit of a finite best-of-shots reduction with a certified maximum."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalResult:
    answer: object
    shots: int
    decision_checks: int


def best_of_shots(draw_batch, ranking, certified_maximum, *, shot_cap, block_size):
    """ranking returns a total-order key or None for an infeasible measurement.

    certified_maximum must certify that no possible continuation can outrank the
    incumbent, including tie-breaking. A caller requiring the histogram cannot use
    this output port. draw_batch is invoked only for blocks actually consumed.
    """
    if type(shot_cap) is not int or type(block_size) is not int or min(shot_cap,block_size)<=0:
        raise ValueError("Supply a finite positive final shot budget and block size")
    used=checks=0
    answer=best=None
    while used<shot_cap:
        request=min(block_size,shot_cap-used)
        observed=list(draw_batch(request))
        if len(observed)!=request:
            raise ValueError("Final instrument returned an incomplete block")
        for label in observed:
            key=ranking(label)
            if key is not None and (best is None or key>best):
                answer,best=label,key
        used+=request
        checks+=1
        if answer is not None and certified_maximum(answer):
            break
    return TerminalResult(answer,used,checks)
