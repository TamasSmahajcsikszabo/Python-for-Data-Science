import sys, timing
reps = 10000
replist = list(range(reps))

# scenario1: abs()
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

for test in (forLoop, listComp, mapCall, genExpr, genFunc):
    (bestof, (total, result)) = timing.bestoftotal(5, 1000, test)
    print('%-9s: %.5f => [%s...%s]' %
          (test.__name__, bestof, result[0], result[-1]))


# scenario2: inline operation on each iteration
def forLoop():
    res = []
    for x in replist:
        res.append(x + 10)
    return res

def listComp():
    return [x + 10 for x in replist]

def mapCall():
    return list(map((lambda x: x + 10), replist))

def genExpr():
    return list(x+10 for x in replist)

def genFunc():
    def gen():
        for x in replist:
            yield x+10
    return list(gen())

for test in (forLoop, listComp, mapCall, genExpr, genFunc):
    (bestof, (total, result)) = timing.bestoftotal(5, 1000, test)
    print('%-9s: %.5f => [%s...%s]' %
          (test.__name__, bestof, result[0], result[-1]))


# scenario3: custom function
import util.supplementary_functions as sf
replist = ['Sentence as this is fairly simple and also is missing meaning',
           'Testing is very important, because time is money']
seps = sf.load_local_vocabulary("separators")

def forLoop():
    res = []
    for x in replist:
        res.append(sf.tokenize_to_sentences(x, seps))
    return res

def listComp():
    return [x + 10 for x in replist]

def mapCall():
    return list(map((lambda x: x + 10), replist))

def genExpr():
    return list(x+10 for x in replist)

def genFunc():
    def gen():
        for x in replist:
            yield x+10
    return list(gen())

for test in (forLoop, listComp, mapCall, genExpr, genFunc):
    (bestof, (total, result)) = timing.bestoftotal(5, 1000, test)
    print('%-9s: %.5f => [%s...%s]' %
          (test.__name__, bestof, result[0], result[-1]))

