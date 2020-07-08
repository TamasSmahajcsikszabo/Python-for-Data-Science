a = 3
b = a
a = 'spam'  # it now points to a string object
b           # still references to the object '3'

a = 3
b = a
a = a + 2
a           # references to a completely new object, namely 5
b           # still references to 3

### mutable objects can be changed
l1 = [1, 2, 3]
l2 = l1
l1[0] = 7   # we changed the object in-place l2 references, too
l2

# if such behaviour is not wanted, copy the object
l1 = [1, 2, 3]
l2 = l1[:]
l1[0] = 8
l2

# alternatives for copying
l3 = list(l1)
import copy
l4 = copy.copy(l1)

# copying other mutable objects (sets and dictionaries) with their x.copy() method call
# OR copy them with dict() and set()

## testing for equality
x = 34
y = x

x == y # tests if the values of the variables are equal
x is y # tests if the variables are referenced to the same object

### counting number of references
import sys
sys.getrefcount(1) # object 1's total reference count