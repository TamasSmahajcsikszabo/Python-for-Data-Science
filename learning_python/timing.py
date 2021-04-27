import time
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    timer = time.perf_counter
else:
    timer = time.clock if sys.platform[:3] == "win" else time.time


def total(func, *pargs, _reps=1000, **kargs):
    """
    Total time to run func() reps time.
    Returns (total time, last result)
    """
    # _reps = kargs.pop('_reps', 1000)
    replist = list(range(_reps))
    start = timer()
    for i in replist:
        ret = func(*pargs, **kargs)
    elapsed = timer() - start
    return (elapsed, ret)


def bestof(func, *pargs, _reps=5, **kargs):
    """
    Quickest func() among reps runs.
    Returns (best time, last result)
    """
    # _reps = kargs.pop('_reps', 1000)
    best = 2**32
    for i in range(_reps):
        start = timer()
        ret = func(*pargs, **kargs)
        elapsed = timer() - start
        if elapsed < best:
            best = elapsed
    return (best, ret)


def bestoftotal(func, *pargs, _reps1=5, **kargs):
    """
    Best of totals:
        (best of reps1 runs of (total of reps2 runs of func))
    """
    # _reps1 = kargs.pop('_reps', 5)
    return min(total(func, *pargs, **kargs) for i in range(_reps1))

# def bestoftotal_t(reps1, reps2, func, *pargs, **kargs):
#     """
#     Tuple expression version of bestoftotal()
#     """
#     return min(total(func, *pargs, **kargs) for i in range(reps1))
