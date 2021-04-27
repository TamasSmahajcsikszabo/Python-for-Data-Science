# numbers
a = 3.1410000000005 * 2
b = print(3.1410000000005 * 2)


import math
math.pi
math.sqrt(121)

import random
random.choice([1,2,3,4])

#strings
t = "This is an example"

len(t)
t[:]
t[5:-1]
t1 = "this"
t2 = "has"
t3 = "been"
text = t1 + " " + t2 + (" " + t3) *2

B = "shrubbery"
L = list(B)
L[1] = "c"
c = ''.join(L)

B = bytearray(b'shrubbery')
B.extend(b' done!')
B_decoded = B.decode()

#string methods
B = B_decoded
B
B.find("done")
B.replace('done', 'failed')
B.split(' ')
B2 = B + " "
B2.rstrip()
B2.rstrip().split(' ')
B2.isdigit()
B2.upper()

# formatting
'%s, eggs and %s' % ('Spanish inquisitor', 'Mr Gumby') # expression
'{0}, eggs and {1}'.format('Spanish inquisitor', 'Mr Gumby') #method
'{}, eggs and {}'.format('Spanish inquisitor', 'Mr Gumby') #method

# formatting in numeric reports
c = 2989657665.4586
'{:,.2f}'.format(c)
'%.2f | %+05d' % (3.14589, -42)

# getting help with the built-in dir function
dir(c)  # this lists the methods and function attributes to the given object

# to get real context use help
t = 'temple'
help(t.replace)

# escape sequences
# \n - end of line
# \t - tab
# \0 - binary zero byte

s = 'A\nB\tC'
print(s)
s2 = 'A\0B\0C'
print(s2)

#multiline
msg = """
asasa
aasasa''''""ASASA'3+
333
"""

### Pattern matching in Pythin with the 're' module
import re
help(re)
match = re.match('Hello[ \0]*(.*)world', 'Hello Python world')
match.group(1)
match.groups()

### Lists
L = [123, 'SPAM', 1.23]
len(L)
L[0]
L[:-1]
L * 2
L + [4,5,6]
L[:-2]
L.append('NI')
L
L.pop(2)            # deletes an item from the list - pop is a method
del L[1]            # deletes an item from the list - del is a statement
# inserting
# removing by value
# extending
L.insert(2, "Holy Hand Granade")
L.remove('NI')
names = ['Sir Lancelot', 'Sir Galahad']
L.extend(names)
# sorting lists
names.sort()
names.reverse()
# nesting lists
nested = [[1,2,3], ["fish", "horse", "pig"]]
# addressing a point in the array
nested[1][2]
### for scientific calculation use NumPy or SciPy

### comprehensions [they result in new list]
names.append('King Arthur')
nested.append(names)
col1 = [row[0] for row in nested]

#filtering
matrix = [[1,2,3],
          [2,3,4],
          [2,4,5]]
col1 = [row[0] for row in matrix if row[0] % 2 == 0]
col1 = [row[0] + 1 for row in matrix if row[0] % 2 == 0] #further operations on the values
#iterating
diag = [matrix[i][i] for i in [0,1,2]]
#ranging
ranged = list(range(-6, 7, 2)) # from - 6 generates 7 items by 2
[[x **2, x**3] for x in range(4)]
[[x, x / 2, x ** 2 ] for x in range(5) if x % 2 == 0]

# comprehensions can be generalized to produce a generator
G = (sum(row) for row in matrix)
next(G)
list(map(sum, matrix)) #map built-in

### more complex dictionaries


# using comprehension syntax to create sets and dictionaries
{sum(row) for row in matrix}
{i : sum(matrix[i]) for i in range(3)}

# variants
help(ord)
help(set)
[ord(x) for x in 'spaam'] # output is list of ordinals of characters
{ord(x) for x in 'spaam'} # a set of the values - duplicates removed!
{x : ord(x) for x in 'spaam'} # a dictionary, keys are unique!
g = (ord(x) for x in 'spaam') # a generator
for i in g:
    print(i)

### dictionaries
# they store objects by unique keys rather than by relative position
D = {'food' : 'Spam', 'quantity' : 4, 'color' : 'pink'} # [] reference references the key
D['food']
D['quantity'] += 1 # dictionaries are mutable!!!

# building a dictionary (unlike lists, here out-of-bounds assignments are possible
E = {}
E['name'] = 'Bob'
E['job'] = 'plumber'
E['age'] = 40

# second way of creating a dictionary with dict type name
E2 = dict(name = 'Bob', job = 'plumber', age = 40)

# third way of creating a dictionary with dict type name + zipping two sequences
E3 = dict(zip(['name', 'job', 'age'], ['Bob', 'plumber', 40]))
# list(zip(['a', 'b', 'c'], [1, 2, 3]))
help(zip)

### complex dictionaries
record = {'name' : {'first' : 'Bob', 'last' : 'Average'},
    'job' : ['plumber', 'mechanic'],
    'age' : 51.4}

#accessing items
record['job'][0]
record['name']['first']
#appending / changing
record['job'].append('janitor')
record['name']['first'] = 'Johnny'

# testing for presence
'name' in record

if 'name' in record:
    print('Found!')

value = record.get('name', 0)
value = record['name'] if 'name' in record else 0

### for & while loops
# impose ordering with:
# 1. keys dictionary method
# 2. sort list method
# 3. for loops

ks = list(record.keys())
ks.sort() # since ks object is a list, it's mutable on its place

for key in ks:
    print(key, '=>', record[key])

    # this is a 3-step method and can be substituted by the built-in 'sorted()' function

for key in sorted(record):
    print(key, '=>', record[key])

    # for loop operates on any item which is a sequence
for c in 'Camelet':
    print(c.upper())

# the while loop is more general

x = 10
while x > 0:
    print('spam! ' * x)
    x -= 1

# iteration protocol - responds to the iter() call
# and raises an exception when finished producing values

# list comprehension example and its loop variant
squares = [ x **2 for x in [1,2,3,4]] # list comprehension

squares = []
for x in [1,2,3,4]:
    squares.append(x ** 2)

# considering run speed:
# for loops are slower
# comprehensions and related tools like map and filter run twice as fast
# if performance testing is needed: time and timeit modules + profile module

### tuples
# immutable lists
# fixed collections of items
T = (1, 2, 3, 4)
T = T + (5, 6, 6)
T[0] = 2
len(T)

# tuple methods
T.index(6)
T.count(6) # number of items

# immutable - no item assignment or append
T[0] = 2
# solution
T = (2,) + T[1:] ### note the TRAILING comma

# mixes types and nesting allowed, but no shrinking / growing
T2 = 1, 2, [3, 4, 5, 5], ['Harry', 4, 5]

### files
# open() function (two argments: external file name + optional processing mode)
f = open('./learning_python/data.txt', 'w')
f.write('Hello \n')
f.write('world! \n')
f.close()

# reading back the file
f = open('./learning_python/data.txt', 'r')
text = f.read() # a file's contents are always string regardless of the type of the file
print(text)
text_list = text.split()

#reading a file line by line with an iterator

for line in open('./learning_python/data.txt', 'r'): print(line)
dir(f)

### binary bytes files
# the struct module creates and reads binary bytes files
import struct
packed = struct.pack('>i4sh', 7, b'psam', 8)
file = open('C:\OneDrive\pycode\data.bin', 'wb')
file.write(packed)
file.close()
data = open('C:\OneDrive\pycode\data.bin', 'rb').read()
list(data)
struct.unpack('>i4sh', data) # turn it back into objects

### unicode text files
# non-ASCII text
t = 'sp\xc4m'

file = open('unicode.txt', 'w', encoding = 'utf-8')
file.write(t)
file.close()

text = open('unicode.txt', 'r', encoding = 'utf-8').read()
print(text)
len(text)

raw = open('unicode.txt', 'rb').read()
raw.decode('utf-8')

### sets (unordered collections of unqiue and immputable objects)
#1. creating from sequence
set1 = set('spam')
set2 = {'c', 'a', 'm', 'e','l','o','t'}

tuple_from_sets = set1, set2

#intersection
set1 & set2
#union
set1 | set2
#difference
set1 - set2
#superset
set2 > set1

#set comprehenson
{x ** 2 for x in [1, 2, 3, 4]}

# purpose of sets
# 1. filtering out duplicates
list(set([1,2,3,4,1,2,3,4]))

# 2. isolating differences
set('spam') - set('ham')

# 3. performing order-neutral equality tests
set('spam') == set('pasm')

# 4. like all other collection types in Python, they support membership tests
'p' in set('spam')

### decimal numbers
1 / 3
import decimal
d = decimal.Decimal('3.142')
decimal.getcontext().prec = 2
decimal.Decimal('1.00') / decimal.Decimal('3.00')

### fraction numbers
from fractions import Fraction
f = Fraction(2,3)

### booleans
# TRUE, FALSE, NONE

### testing for type
L = [1, 3, 98]

#1. testing for the type itself
if type(L) == type([]):
    print('Yes!')

#2. using the type name
if type(L) == list:
    print('Yes!')

#3. the Object-Oriented (OO) way
if isinstance(L, list):
      print('Yes!')

# normally this is not the Pythonian way!

### user-defined classes
# extending the standard core set of items in Python

class Worker:
    def __init__(self, name, pay):
        self.name = name
        self.pay = pay
    def lastName(self):
        return self.name.split()[-1]
    def giveRaise(self, percent):
        self.pay *= (1.0 + percent)

pete = Worker('Peter Harrow', 50000)
pete.lastName()
pete.giveRaise(0.2)
pete.pay
