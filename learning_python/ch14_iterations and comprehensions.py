print(open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py').read())
open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py').read()

f = open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')
f.__next__() ## a method with a stopiteration exception = an iterator

for line in open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py'):    # it does not read in the full file into memory
    print(line, end = '')

for line in open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py').readlines(): # This reads in the full file into memory
    print(line, end = '')

# iteration protocol:
# 1. iterable - the iter built-in calls __iter__ on it
# 2. iterator - obtained by the iterable, the next built in calls __next__ on it and returns stopiteration exception at the end

keys = ['spam', 'ham','ni']
values = range(1,4)

D = {}
for(k, v) in zip(keys, values): D[k] = v

### dictionary iteration - key method no longer needed in 3.x
for keys in D.keys():
    print(keys, D[keys])
# the new method:
for key in D:
    print(key, D[key])

## List comprehensions -  they are run at C language speed, loops are not
L = [1,2,3,4,5,6]

for i in range(len(L)):
    L[i] = L[i] + 10
# a simple comprehension instead
L = [x+10 for x in L]

# in a more verbose way
res = []
L = [1,2,3,4,5,6]
for x in L:
    res.append(x + 10)

# list comprehensions in action:
f = open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')
lines = f.readlines()
L = [line.rstrip() for line in lines]

## to speed it up by avoiding readlines():
lines = [line.rstrip() for line in open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')]

## adding filtering
lines = [line.rstrip() for line in open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py') if line[0] == 'p']
lines = [line.rstrip() for line in open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py') if line.rstrip()[-1].isdigit()]

## nested loops
[x + y for x in 'abc' for y in 'clm']

## others
list(map(str.upper, open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py'))) ### it returns an iterable object so list call is needed

#other examples
sorted(open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py'))
list(zip(open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py'), open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py'))) ## the only one here which returns a list not an iterable in 3.x
list(enumerate( open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')))
list(filter(bool,   open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')))
import functools, operator
functools.reduce(operator.add,  open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py'))

# these follow the iteration protocol:
list(open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py'))
tuple(open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py'))
'&&'.join(open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py'))
a,b,c,d = open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')
a, *b = open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')
'y = 2\n' in open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')
L = [11,12,13,14]; L[1:3] = open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')
L.extend(open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')) # extend iterates automatically
# append does not:
L = [11]; L.append(open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')); list(L)
set(open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py'))
{line for line in open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py')}
{x: line for x, line in enumerate(open('C:\\Users\\tamas\\OneDrive\\python_code\\script2.py'))}
r = [1,2,3,4,5]; sum(r); min(r); max(r)
t = ['spam', 'ni', '']; any(t); all(t)
# *arg for unpacking
x = (1,2); y = (3,4); list(zip(x,y))
A, B = zip(*zip(x,y))

### THE range ITERABLE # it's not its own iterators, so it allows multiple iterators in it
r = range(0,20)
list(r)
len(r)
list(r[5:-1])

### THE map, zip and filter ITERABLES # they are their own iterators
m = list(map(abs, (-1,0,1))); m
z = zip((1,2,3),(4,5,6),(7,8,9)); list(z)
[print(pair) for pair in zip((1,2),(3,4))]

## DICTIONARY keys, values and item methods - ITERABLE VIEW OBJECTS
# they peoduce one item at a time
D = dict(a = 1, b = 2, c = 3)
for k in D.keys(): print(k, end = '')
# if put into list(), they can be printed, sliced etc.
v = D.values(); list(v)[:2]
list(D.items())
# dictionaries are iterables themselces, the iterators returns successive keys
