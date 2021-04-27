# argument / parameter passing

import sys
import builtins
x = 1
y = [1, 2, 3]


def changer(a, b):
    a = 2
    b[0] = 'spam'


changer(x, y)

# mutable arguments are impacted by the function and change the caller
# mutable arguments are input to and sometimes output from the function

# avoiding in-place changes with list copies
L = [1, 2, 3]

changed = changer(x, L[:])

# avoiding in-place changes within the function


def changer2(a, b):
    a = 2
    b = b[:]
    b[0] = 'spam'
    return (a, b)


x = 1
y = [1, 2]
mix = changer2(x, y)

# COLLECTING ARGUMENTS AT HEADER LEVEL
# * positional
# ** keyword

# *args collects all unmatched positional arguments into a tuple


def f(*args):
    print(args)


f(1, 2, 3, 4, 5)

# **args collect all keyword assignemnts into a dictionary


def f(**args):
    print(args)


f(a=1, b=2)

# * behaves in the reverse way in function calls


def f(a, b, c, d): print(a, b, c, d)


args = (1, 2)
args += (3, 4)
f(*args)  # * unpacks the tuple

k = ['a', 'b', 'c']
v = (1, 2, 3)
D = {k: v for k, v in zip(k, v)}
args = D
args['d'] = 4
f(**args)  # it unpacks the dictionary's values


def tracer(func, *pargs, **kargs):
    print('calling:', func.__name__)
    return func(*pargs, **kargs)


def func(a, b, c, d):
    return a + b + c + d


tracer(func, 1, 2, c=3, d=4)

# keyword only arguments
# they appear after *args


def kwonly(a, *b, c):
    print(a, b, c)


kwonly(1, 2, c=3) # notice tuple for *parg

# alternatively


def kwonly2(a, *, b, c):
    print(a, b, c)


kwonly2(1)
# they can have defaults, which turns them into optional
# **args is always the ending parameter


def min1(*args):
    res = args[0]
    for arg in args[1:]:
        if arg < res:
            res = arg
    return res


f = 1, 2, 3, 4, 5
min1(f)


def min2(first, *rest):
    for arg in rest:
        if arg < first:
            first = arg
    return first


f2 = [3, 2, 1]
print(min2(f2))


def min3(*args):
    tmp = list(args)
    tmp.sort()
    return tmp[0]


min3(f)
min3([1, 1], [2, 2])
min1(3, 4, 1, 2)
min2(2, 3, 4)


def minmax(test, *args):
    res = args[0]
    for arg in args[1:]:
        if test(arg, res):
            res = arg
    return res


def MAX(x, y): return x > y
def MIN(x, y): return x < y


minmax(MAX, 1, 2, 3, 4, 4, 5)
minmax(MIN, 4, 5)


x = 'spam'
y = 'ham'
intersect = [i for i in x if i in y]


def intersect(*args):
    res = []
    for i in args[0]:
        if i in res: continue
        for other in args[1:]:
            if i not in other: break
        else:
            res.append(i)
    return res


intersect((1, 2, 3, 4), (3, 4, 5, 6))


def union(*args):
    res = []
    for seq in args:
        for i in seq:
            if i not in res:
                res.append(i)
    return res


union((1, 2, 3, 4), (3, 4, 5, 6))


def tester(func, items, trace=True):
    for i in range(len(items)):
        items = items[1:] + items[:1]
        if trace: print(items)
        print(sorted(func(*items)))


tester(intersect, ((1, 2, 3, 4), (3, 4, 5, 6), (5, 6, 1)))


def print3(*args, **kargs):
    sep = kargs.get('sep', ' ')
    end = kargs.get('end', '\n')
    file = kargs.get('file', sys.stdout)
    output = ''
    first = True
    for arg in args:
        output += ('' if first else sep) + str(arg)
        first = False
    file.write(output + end)


print3(4, 5, 6, sep='_', end='_*')

# with keyword-only arguments:


def print3_alt(*args, sep=' ', end='\n', file=sys.stdout):
    output = ''
    first = True
    for arg in args:
        output += ('' if first else sep) + str(arg)
        first = False
    file.write(output + end)


print3_alt(4, 5, 6)


def print3_alt2(*args, **kargs):
    sep = kargs.pop('sep', ' ')
    end = kargs.pop('end', '\n')
    file = kargs.pop('file', sys.stdout)
    if kargs: raise TypeError('extra keywords: %s' % kargs)
    output = ''
    first = True
    for arg in args:
        output += ('' if first else sep) + str(arg)
        first = False
    file.write(output + end)

 # keyword-only arguments play important role in tkinter
 from tkinter import *
 widget = Button(text='Press me'), command =someFunction)

 # Quiz

def func(a, b=4, c=5):
     print(a,b,c)
func(1,2)

def func(a, *pargs):
    print(a, pargs)
func(1,2,3)

def func(a, **kargs):
    print(a, kargs)
func(a=1, b=2, c=3)

def func(a, b, c = 3, d = 4): print(a,b,c,d)

func(4, *(5,6))

def func(a,b,c): a = 2; b[0] = 'x'; c['a']= 'y'

l = 1
m = [1]
n = {'a' : 0}

func(l,m,n)
