l = 'spam'
c = list(map(ord, l)) # this applies a funtion over an iterable
# list comprehension evaluate an expression over an iterable
l2 = [ord(x) for x in l]

[x ** 2 for x in range(10)]
list(map((lambda x: x ** 2), range(10)))

list(filter((lambda x: x % 2 == 0), range(10)))
[x for x in range(10) if x % 2 == 0]

[x ** 2 for x in range(10) if x % 2 == 0]

list(map((lambda x: x ** 2),filter((lambda x: x % 2 == 0), range(10))))

# list comprehensions can have any number of nested for parts
[x + y for x in [1,2,3] for y in [10, 20, 30]]
# each for section can have its own if conditions

[(x, y) for x in range(10) if x % 2 ==0 for y in range(10) if y % 2 == 1]

res = []
for x in range(10):
    if x % 2 == 0:
        for y in range(10):
            if y % 2 == 1:
                res.append((x,y))
print(res)

## coding matrix structures with list comprehensions
M = [[1,2,3],
     [4,5,6],
     [7,8,9]
    ]

M[0][2]

[row[1] for row in M]
[M[i][i] for i in [0,1,2]]
len(M)

L = [[1,2,3], [4,5,6]]

for i in range(len(L)):
    for j in range(len(L[i])):
        L[i][j] += 10

[[col + 10 for col in row] for row in M]

res = []
for row in M:
    tmp = []
    for col in row:
        tmp.append(col + 10)
    res.append(tmp)

M = [[1,2,3],[4,5,6],[7,8,9]]
N = [[2,2,2],[3,3,3],[4,4,4]]

[[M[row][col] * N[row][col] for col in range(3)] for row in range(3)]

[[col1 * col2 in (col1, col2) in zip(row1, row2)] for (row1, row2) in zip(M, N)]

res = []
for (row1, row2) in zip(M, N):
    tmp = []
    for (col1, col2) in zip(row1, row2):
        tmp.append(col1 * col2)
    res.append(tmp)

list(zip(M,N))

row1 = [1,2,3]
row2 = [2,2,2]

list(zip(row1, row2))

### map and list comprehensions run at C code level
# Python's standard SQL API returns results like this which can benefit from map and LC

sqlresult = [('sue', 40, 'female'), ('tom', 42, 'male')]

[name for (name, age, sex) in sqlresult]
list(map((lambda record: record[1]), sqlresult))


### GENERATOR EXPRESSIONS
## their purpose is to retun results when needed, not all at once
## GENERATOR FUNCTIONS: def + yield, suspends and resumes its state between calls
## GENERATOR EXPRESSIONS: just like LC, but returns an object which produces results on demand
## yield suspends and resumes the function run enabling collecting data from various points of time
## they run until yield or StopIteration exception

## GENERATOR FUNCTIONS
def gensquares(n):
    for i in range(n):
        yield i ** 2

for i in gensquares(5):
    print(i)

def ups(line):
    for sub in line.split(','):
        yield sub.upper()

list = ['aaa','bbb','cCc']
[next(ups(i)) for i in list]
{c: next(ups(i)) for (c, i) in enumerate(list)}

### the send method
def gen():
    for i in range(10):
        x = yield i
        print(x)

G = gen()
next(G)

G.send(7) # send can be used to send a termination code to the generator or redirecting it

## GENERATOR EXPRESSIONS:
# enclosed in parenthesis like tuples
list(x ** 2 for x in range(4))
''.join(x.upper() for x in ['aaa','bbb','ccc'])
''.join(x.upper() for x in 'aaa,bbb,ccc'.split(','))
# in combination with tuple assignment
a,b,c = (x + '\n' for x in 'aaa,bbb,ccc'.split(','))
## just like GEN functions, they are optimizations, as they don't require the full result set at once
## GENCs are slower than LCs, so they are optimal when large data is processed and only some of the results
## are needed

numbers = [-3, -2, 0, 4, 5]
list(map(abs, numbers))
list(abs(x) for x in numbers)

list(map(lambda x: x ** 2 + 1, numbers))
list(x**2 + 1 for x in numbers)

line = 'abb,e,fgh,i'

''.join([i.upper() for i in line.split(',')])
''.join(map(str.upper, line.split(',')))
''.join(i.upper() for i in line.split(','))

''.join(i * 2 for i in line.split(','))
''.join(map(lambda x: x * 2, line.split(',')))

[x ** 2 for x in [abs(x) for x in [-3,-2,-1]]]                  # this makes two lists,     
list(map(lambda x: x ** 2, map(abs, [-3, -2, -1])))             # these two options only generates integers
list(x ** 2 for x in (abs(x) for x in [-3, -2, -1]))

# filtering
line = 'aa bbb ccccc d'
''.join(x for x in line.split(' ') if len(x) > 2)
''.join(filter(lambda x: len(x) > 2, line.split()))

''.join(x.upper() for x in line.split() if len(x) > 2)
''.join(map(str.upper, filter(lambda x: len(x) > 2, line.split())))

## the yield from syntax
def both(n):
    yield from range(n)
    yield from (x ** 2 for x in range(n))

' : '.join(str(i) for i in both(6))
# this allows multiple subgenerators

import os
for (root, subs, files) in os.walk('.'):
    for name in files:
        if name.startswith('call'):
            print(root, name)

D = dict(a = 'bob', b = 'john', c ='julia')

def f(a,b,c): 
    print('%s, %s and %s' % (a,b,c))S

f(*D)
f(**D)
print(*(x.upper() for x in 'spam'))

# iterables in classes
L = 'spam'

for i in range(len(L)):
    L = L[1:] + L[:1]
    print(L,end = ' ')

for i in range(len(L)):
    L = L[i:] + L[:i]
    print(L,end = ' ')

def scramble(x):
    res = []
    for i in range(len(x)):
        res.append(x[i:] + x[:i])
    print(res)

scramble('spam')

def scramble(x):
    return [x[i:] + x[:i] for i in range(len(x))]
scramble('spam')
# this is a great version, but not memory efficient
# also forces the caller to wait until the entire list is complete

## optimize it with turning it into generator funcion
## doing the same work but separating it into smaller time sloces
def scramble(x):
    for i in range(len(x)):
        yield x[i:] + x[:i]

list(scramble('spam'))

x = 'spam'
scramble = (x[i:] + x[:i] for i in range(len(x))) # in this format cannot contain statement
list(scramble)

# to contain statement, wrap it into a simple function
scramble = lambda seq: (seq[i:] + seq[:i] for i in range(len(seq)))
list(scramble(x))
list(scramble('camelot'))

### permutations
seq = [1,2,3]
def permute1(seq):
    if not seq:
        return [seq]
    else:
        res = []
        for i in range(len(seq)):
            rest = seq[:i] + seq[i+1:]
            for x in permute1(rest):
                res.append(seq[i:i+1] + x)
        return res

permute1([1,2,3])

def permute2(seq):
    if not seq:
        return [seq]
    else:
        res = []
        for i in range(len(seq)):
            rest = seq[:i] + seq[i+1:]
            for x in permute1(rest):
                yield seq[i:i+1] + x

list(permute2([1,2,3]))

def reader(f):
    for l in open(f):
        yield print(l, end = '\n')

file = 'C:/Users/tamas/OneDrive/python_code/1861 First Inauguration Address.txt'
speech  = list(reader(file))

### emulating map

def mymap(func, *seqs):
    res = []
    for args in zip(*seqs):
        res.append(func(*args))
    return res

a = [1,2,3]
b = [4,5,6]
list(mymap(pow, a, b))

def mymap(func, *seqs):
    return [func(*args) for args in zip(*seqs)]
list(mymap(pow, a, b))

# turn this into generators

def mymap(func, *seqs):
    for args in zip(*seqs):
        yield func(*args)

list(mymap(pow, a, b))

def mymap(func, *seqs):
    yield [func(*args) for args in zip(*seqs)]
list(mymap(pow, a, b))

### emulating zip

def myzip(*seqs):
    seqs = [list(s) for s in seqs]
    res = []
    while all(seqs):
        res.append(tuple(s.pop(0) for s in seqs))
    return res

a = [1,2,4]
b = [5,7,8]
myzip(a,b)

def mymappad(*seqs, pad = None):
    seqs = [list(s) for s in seqs]
    res = []
    while any(seqs):
        res.append(tuple((s.pop(0) if s else pad) for s in seqs))
    return res

x = [4,5,6]
y = [1,2]
mymappad(x,y)

def myzip(*seqs):
    seqs = [list(s) for s in seqs]
    while all(seqs):
      yield tuple(s.pop(0) for s in seqs)

def mymappad(*seqs, pad = None):
    seqs = [list(s) for s in seqs]
    while any(seqs):
      yield  tuple((s.pop(0) if s else pad) for s in seqs)

list(myzip(x,y))
list(mymappad(x,y))

def myzip(*seqs):
    minlen = min(len(s) for s in seqs)
    return (tuple(s[i] for s in seqs) for i in range(minlen)) 
list(myzip(x,y))

### set and disctionary comprehensions vs. generators
{x: x**2 for x in range(10)}
{x**2 for x in range(10)}
set(x**2 for x in range(10))
dict((x, x**2) for x in range(10))

##x nested loops and duplicate rules
[x + y for x in [1,2,3] for y in [4,5,6]]
{x + y for x in [1,2,3] for y in [4,5,6]}
{x : y for x in [1,2,3] for y in [4,5,6]}
