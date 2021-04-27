import time
def timer(func, *args):
    start = time.perf_counter()
    for i in range(1000):
        func(*args)
    return time.perf_counter() - start

timer(pow, 2, 1000)
timer(str.upper, 'spam')
timer.total(1000, pow, 2, 1000)[0]
timer.bestof(1000, str.upper, 'spam')
timer.bestoftotal(50,1000,str.upper, 'spam')


## timing script
import sys, timer
reps = 10000
replist = list(range(reps))

def forLoop():
    res = []
    for x in replist:
        res.append(abs(x))
    return res

def listComp():
    return [abs(x) for x in replist]

def mapCall():
    return list(map(abs, replist))

def genExpr():
    return list(abs(x) for x in replist)

def genFunc():
    def gen():
        for x in replist:
            yield abs(x)
    return list(gen())

print(sys.version)
for test in (forLoop, listComp, mapCall, genExpr, genFunc):
    (bestof, (total, result)) = timer.bestoftotal(5, 1000, test)
    print('%-9s: %.5f => [%s...%s]' %
          (test.__name__, bestof, result[0], result[-1]))



# timeit module
import timeit
timeit.repeat() ## combined with min() gives the best time of run
min(timeit.repeat(stmt = "[x**2 for x in range(1000)]", number = 1000, repeat = 5))

import chessboard
chessboard.chessboard()
min(timeit.repeat(chessboard.chessboard2(1000), number = 1000, repeat = 5))
