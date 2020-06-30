L = []
for n in range(1000):
    L.append(n ** 2)
print(L)

# IPython In and Out objects
import math
math.sin(2)
math.cos(2)
print(In)
print(Out)

# suppress output and not including it in the Out object with semicolon
a = 14 + 1;
a in Out

%history -n 1-4

# shell commands in IP with !
!echo "Hello"
!ls
!pwd

var = "This is a Python object"
!echo {var} # {} is used to reference Python objects

# embedding magic Shell commands in IPython
# returns a specifict IPython type
contents = !ls
print(contents)
type(contents)

# toggle automagic functions
%automagic

pwd
ls
cat chapter1.py
mkdir tmp
cd tmp
rm -r tmp
%cd ..

## errors and debugging
# exception mode toggle
%xmode

def func(a,b,):
    return a / b

def func2(x):
    a = x
    b = x-1
    return a / b


%xmode Minimal
func2(1)

# debuggin
# the standard Python debuggin tool is pdb
# Ipyton has its own ipdb
func2(1)
%debug

# profiling and timing code
%timeit sum(range(1,100))

%%timeit
total = 0
for i in range(1000):
    for j in range(1000):
        total += i ** j

# the effect of sorting
import random
L = [random.random() for i in range(1000)]
%timeit L.sort()

L = [random.random() for i in range(1000000)]
print('execution time for unsorted list')
%time L.sort()

print('execution time for an already sorted list')
%time L.sort()

%%time
total = 0
for i in range(1000):
    for j in range(1000):
        total += i ** j

### profiling full scipts with %prun

def sum_of_lists(N):
    total = 0
    for i in range(5):
        L = [i ^ (j >> i) for j in range(N)]
        total += sum(L)
    return total

%prun sum_of_lists(100000)

# line-by-line profiling
import line_profiler
%lprun - f sum_of_lists() sum_of_lists(100000)

# profiling memory use
import memory_profiler
%load_ext memory_profiler # loads the IPython extension

%memit sum_of_lists(10000000)
#the line-by-line alternative works only on modules
%%file sum_of_lists.py
def sum_of_lists(N):
    total = 0
    for i in range(5):
        L = [i ^ (j >> i) for j in range(N)]
        total += sum(L)
    return total
# we created a module from the function
from sum_of_lists import sum_of_lists
%mprun -f sum_of_lists sum_of_lists(10000)

