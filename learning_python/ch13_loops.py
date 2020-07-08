# while statement

def slicer():
    x = input(str('Give me a string!'))
    while x:
        print(x, end = ' ')
        x = x[1:]

def counter():
    inputv = list(input('Give me two numbers'))
    a = float(inputv[0])
    b = float(inputv[1])
    while a < b:
        print(a, end = ' ')
        a += 1

# pass statement = NULL as an object; as an alternative use ... (ellipses)
# continue jumps to the top of the loop
def countdown():
    x = int(input('give me a number'))
    while x:
        x -= 1
        if x % 2 != 0: continue
        print(x, end = ' ')

# break is an immediate exit from the loop

while True:
    name = str(input('What is your name?'))
    if name == 'stop': break
    print('Hello', name, '!')

y = 100
x = y // 2
while x > 1:
    if y % x == 0:
        print(x, 'has factor', y)
        break
    x -= 1
else:
        print(y, 'is prime')

## for loops
for x in ['ham', 'ni', 'spam']:
    print(x, end = ' ')

sum = 0
for x in [1,2,3,4,5,6]:
    sum = sum + x
else:
    print(sum)

prod = 1
for x in [1,2,3,4]:
     prod *= x
else: print(prod)

for c in 'lumberjack':
    print(c, end = ' ')

T = ('I', 'am', 'okay')
for i in T:
    print(i, end = ' ')

# tuple unpacking with for loop
T = [(1,2), (3,4), (5,6)]
for (a,b) in T:
    print(a,b)

D = {'a' : 1, 'b' : 2, 'c': 3}
for (key, value) in D.items():
    print(key, '=>', value)

T = [(1,2,4,5,6,7,8), (1,2,4,5,6,7,8)]
for (a, *b, c) in T:
    print(a,b,c)    ### starred reference always returns LISTS

### nested for loops
items = [3,4,5,6,3.14]
keys = [3.14, 2.23] 

for k in keys:
    for i in items:
        if i == k: 
            print(k, "was found!")
            break
    else:
            print(k, "was not found!")

for k in keys:
    if k in items:
        print(k, "was found!")
    else:
        print(k, "was not found!")

seq1 = 'scam'
seq2 = 'spam'
res = []

for i in seq1:
    if i in seq2:
        res.append(i)
[i for i in seq1 if i in seq2]

for char in open('C:\\Users\\tamas\\OneDrive\\python_code\\message.txt').read():
    print(char, end = '')

file = open('C:\\Users\\tamas\\OneDrive\\python_code\\message.txt')

while True:
    line = file.readline()
    if not line: break
    print(line.rstrip())

# open in chunks
while True:
    line = file.read(10)
    if not line: break
    print(line)

# read in by lines - it does not load the entire file into the memory at once
for line in open('C:\\Users\\tamas\\OneDrive\\python_code\\message.txt'): 
    print(line.rstrip()) #rstrip() removes \n from the end of the lines

# specific looping techniques
# range - it creates an iterable which needs to be wrapped in a constructor to get the desired object type; but for loop forces it, so no wrapper is needed
list(range(0, 20, 2)) # step defaults to 1
list(range(10, -11, -1)) # the second argument is non-inclusive

for i in range(1,6,1):
    if i < 2:
        print(i, "Python")
    else:
        print(i, "Pythons")

t = 'Camelot'
for i in range(len(t)): print(t[i], end = ' ')
for i in t: print(i, end = ' ')

t = 'camelot'
for i in range(len(t)):
    x = t[i:] + t[:i]
    print(x, end = ' ')

L = list(range(1,11,1))
for i in range(len(L)):
    L[i] += 1
    print(L[i], end = ' ')
[x + 1 for x in L] # a list comprehension is quicker, but does not change the list in place

# zip - parallel traversal
# it creates tuple pairs or triplet and so on
# a wrapper is needed, just like for range
L1 = [1,2,3,4,5,6]
L2 = [7,8,9,10,11,12]

for (a,b) in zip(L1, L2):
    print(a, b, '--', a+b)
# zip truncates to the shortest input when more are present
# dictionary creation with zip
keys = ['spam', 'ham','ni']
values = range(1,4)

D = {}
for(k, v) in zip(keys, values): D[k] = v
D2 = dict(zip(keys, values))
{k: v for (k, v) in zip(keys, values)}

## enumrate
# for dual usage modes

# the old way:
t = 'samp'
c = 0
for i in t:
    print(t, 'appears at offset', c)
    c += 1

# the new way:
for (offset, item) in enumerate(t):
    print(item, 'appears at offset', offset)
# enumerate return a generator object, whose elements can be extracted with the next method:
e = enumerate(t)
next(e)

import os
F = os.popen('dir')
F.readline()
F.read(100)

for line in os.popen('dir'):
    print(line.rstrip())

for line in os.popen('systeminfo'):
    print(line.rstrip())

for line in os.popen('systeminfo'):
    parts = line.split(':')
    if parts and parts[0].lower() == 'system type':
        print(parts[1].strip())

from urllib.request import urlopen
for line in urlopen('http://learning-python.com/books'):
    print(line)