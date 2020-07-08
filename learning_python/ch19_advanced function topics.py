### COHESION - decomposing a task into functions
            # one function, one purpose


### COUPLING - how functions communicate
            # no global vaiables
            # no change in input

### recursive functions
# route planning, language analysis

def mysum(L):
    print(L)
    if not L:
        return 0
    else:
        return L[0] + mysum(L[1:])

n = (1,2,3,4,5)
mysum(n)

def mysum(L):
    return 0 if not L else L[0] + mysum(L[1:])
def mysum(L):
    return L[0] if len(L) == 1 else L[0] + mysum(L[1:])
def mysum(L):
    first, *rest = L ## extended sequence assignment
    return first if not rest else first + mysum(rest) 
mysum('spam')

## these are examples for direct recursion

## INDIRECT recursion:
def mysum(L):
    if not L: return 0
    return nonempty(L)

def nonempty(L):
    return L[0] + mysum(L[1:])

mysum([1.1, 2.2, 3.3, 4.4, 5.5])

## RECURSION is good for nonlinear iteration
L = [1,2,3,4, [1,2,3], [[12,12], 4,5,6]]
# depth can be arbitrary

def sumtree(L):
    tot = 0
    for i in L:
        if not isinstance(i, list):
            tot += i
        else:
           tot += sumtree(i)
    return tot

sumtree(L)

def sumtree2(L):
    tot = 0
    items = list(L)
    while items:
        front = items.pop(0)
        if not isinstance(front, list):
            tot += front
        else:
            items[:0] = front
    return tot

sumtree2(L)

## checking and setting recursive deph
import sys
sys.getrecursionlimit()
sys.setrecursionlimit(10000)

## first class object model in Python
## adding user-defined attributes to functions
func.count = 1

### annotations # only for def
def func(a: int = 1, b: 'spam' = 4) -> int:
    return(a + b)

for arg in func.__annotations__:
    print(arg, "=>", func.__annotations__[arg])

func()

### lambda (it's an expression, def is a statement)
f = lambda a, b, c : a + b + c
f(1,4,5)

f = lambda a = 'fie', b = 'foe', c = 'foo': a + b + c
f("wee")

def knights():
    title = "Sir"
    full = (lambda x: title + ' ' + x)
    return full

act = knights()
act("Robin")

# it's good for jump tables / action tables
L = [lambda x: x ** 2,
     lambda x: x ** 3,
     lambda x: x ** 4]

L[1](2)

key = 'degree2'
{'degree2' : (lambda x: x ** 2),
 'degree3':(lambda x: x ** 3),
'degree4': (lambda x: x ** 4)}[key](2)

lower = (lambda x,y: x if x < y else y)
lower(10,2)

import sys
showall = lambda x: list(map(sys.stdout.write, x))
showall(['spam\n', 'ham\n'])

showall = lambda x: [sys.stdout.write(line) for line in x]
t = showall(('bright\n', 'side\n', 'of\n', 'life\n'))

((lambda x: (lambda : y, x + y))(99))(4)

## embed function call with lambda into a tkinter button creation call
import sys
from tkinter import Button, mainloop
x = Button(
    text = "Press me",
    ### lambda defers the execution until clicking, not when call creation
    command = (lambda: sys.stadout.write('Spam\n'))) 
x.pack()
mainloop()


### map
counters = [1,2,3,4,5]
list(map(lambda x: x + 3, counters))

def mymap(func, x):
    res = []
    for i in x:
        res.append(func(i))
    return res

def inc(x): return x + 3
list(mymap(inc, counters))
list(map(inc, counters))

## filter
list(filter(lambda x: x > 0, range(-5, 5)))
[x for x in range(-5, 5) if x >0]

## reduce x returns one value
from functools import reduce
reduce((lambda x,y : x * y), [1,2,3])

def myreduce(func, x):
    tally = x[0]
    for next in x[1:]:
        tally = func(tally, next)
    return tally

myreduce((lambda x, y : x + y), [1,2,3])

def myreduce(func, x, dv = 'empty input'):
    if len(x) == 0: return dv
    else:
        tally = x[0]
        for next in x[1:]:
            tally = func(tally, next)
        return tally

myreduce((lambda x, y : x + y), [1,2,3])
myreduce((lambda x, y : x + y), [])