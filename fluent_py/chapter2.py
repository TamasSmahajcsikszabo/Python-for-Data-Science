import collections
import array

# By type:
# container sequences - mutliple types
# obejcts are separate Python objects with their own references
list()
tuple()
collections.deque()

# flat sequences - one type
# values are stores in its own memory space, not as distint objects
str()
bytes()
bytearray()
memoryview
array.array()

# By mutability:
# mutables inherit all methods from immutables

# mutable sequences
list()
bytearray()
array.array()
collections.deque()
memoryview

#immutable sequences
tuple()
str()
bytes()

# superclasses:
# 1. collection and 2. reversed
# 3. sequence
# 4. mutablesequence

# creating sequences (these have their own local scope)
# - lists with listcomps
# - others with genexps (generator expressions)

# list comprehensions
symbols = '$¢£¥€¤'
codes = []
for sym in symbols:
    codes.append(ord(sym))
codes = [ord(sym) for sym in symbols]

