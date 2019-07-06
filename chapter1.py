L = []
for n in range(1000):
    L.append(n ** 2)

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
        total += i * (-1) ** j