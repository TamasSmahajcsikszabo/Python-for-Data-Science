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

