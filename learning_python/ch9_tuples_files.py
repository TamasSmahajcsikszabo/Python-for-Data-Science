(1, 2) + (5, 9)
T = (1, 7) * 4
T[1:5]

T = (1,2,3,4,5)
[x + 20 for x in T]
T.index(1)
T.count(1)

T = (1, 2, 3, [4,5,6])  # tuples are immutable only on the top level
T[3][1] = 7             # mutable elements are still mutable in a tuple like a nested list

from collections import namedtuple
Rec = namedtuple('Rec', ['name', 'age', 'jobs'])        # making the generated class    
bob = Rec('Bob', age = 30, jobs = ['dev', 'mgr'])       # create the named tuple

# files
f = open('C:\OneDrive\pycode\data.txt', 'w')
f.write('Hello \n')
f.write('world! \n')
f.close()

myfile = open('file.txt', 'w')
myfile.write('This is the first line \n')
myfile.write('This is the second line \n')
myfile.close()

myfile = open('file.txt', 'r')
myfile.readline()       # the third is empty line
print(open('file.txt').read())
for line in open('file.txt'):
    print(line, end = '')

X, Y, Z = 1, 2, 3
S = 'spam'
D = {'a' : 1, 'b' : 2}
L = [1,2,3]
F= open('datafile.txt', 'w')
F.write(S + '\n')
F.write('%s, %s, %s\n' % (X, Y, Z))
F.write(str(L) + ' $ ' + str(D) + '\n')
F.close()

chars = open('datafile.txt').read()
print(chars)

F = open('datafile.txt')
line = F.readline().rstrip()
line2 = F.readline().rstrip().rsplit(',')
line2 = [int(i) for i in line2]
line3 = F.readline().split('$')
line3 = [eval(p) for p in line3]        # eval constructs objects from string

# PICKLE module 
# storing any Python objects in files without necessary conversions
D = {'a' : 11, 'b' : 3}
F = open('datafile.pkl', 'wb')          # note the 'wb' mode ~ binary
import pickle
pickle.dump(D, F)
F.close()

F = open('datafile.pkl', 'rb')
E = pickle.load(F)
E
# shelve is also an alternative
import shelve

# JSON
import json
name = dict(first = 'Bob', last = 'Smith')
rec = dict(name = name, job = ['dev', 'mgr'], age = 40.5)
json.dumps(rec)
S = json.dumps(rec)
O = json.loads(S)
O == rec
S == rec # FALSE, because S is a JSON object not a Python dictionary

# from and to file
json.dump(rec, fp = open('testjson.txt', 'w'), indent = 4)
print(open('testjson.txt').read())
P = json.load(open('testjson.txt'))

# CSV
import csv

# struct for handling packed binary data
import struct
# constructing packed binary data
F = open('data.bin', 'wb')
data = struct.pack('>i4sh', 7, b'spam', 8)
F.write(data)
F.close()

# reading back and unpacking
F = open('data.bin', 'rb')
data = F.read()
values = struct.unpack('>i4sh', data)

# Copying a deep data structure
import copy
X = copy.deepcopy(Y)

# recursive comparison

## exercises
tuple = (4,5,6)
modified_tuple = (1, tuple[1], tuple[2])
modified_tuple_2 = (1,) + tuple[1:]

s = 'spam'
l = list(s)
l[1] = 'l'
str = ''.join(l)
str2 = s[0] + 'l' + s[2:]

data = {'name' : {'first' : 'Tamas', 'second':'Szabo'},
        'age': 34,
        'phone' : [0,0,3,6,7,0,4,1,7,7,6,7,7]}

f = open("C:\\Users\\tamas\\OneDrive\\python_code\\data.txt", 'w')
f.write('Hello file world!')
f.close()

f = open('C:\\Users\\tamas\\OneDrive\\python_code\\data.txt', 'r').read()
f