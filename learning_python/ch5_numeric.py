### integer numbers
I = 123
# converters
hex(I)
oct(I)
bin(I)
# complex number
complex(123, 2) # real + imaginary part

# BASIC operations
a = 12
b = 3
a + b

# some advanced operations
b = b + 2
a // b 
a % b
c = 4.0
complex(a,b) + c
4.02 > 4

d = a / (b + c)
print(d)

# display formatting
'%e' % d
'%4.2f' % d
'{0:4.2f}'.format(d)

print(d)
repr(d) # prints as if code

a > b > d < c

#floor vs. truncating
import math
math.floor(2.5)
math.floor(-2.5)

math.trunc(2.5)
math.trunc(-2.5)

## the eval() function treats input as if it were Python code
eval('16')

# converting integers to base-specific strings with string formatting method calls and expressions
'{1:X}, {2:b}'.format(64, 64)

## bitwise operations
x = 1
x << 2 # shift left
x | 2
x & 1

x = 0b001
x << 2
bin(x << 2)
bin(x | 0b010)
bin(x & 0b01)

# bit length with bit.length() method
x = 99
(bin(x), len(bin(x)) - 2, x.bit_length())

# other functions and modules for numeric operations
pow(2,2) # powers numbers
abs(-17*2/4)

import math
# constants
math.pi
math.e
math.sin(12)
math.tan(2)
math.sqrt(14)
min(1,3,4,5,6)
math.trunc(-3.4)
math.floor(-3.4)
round(-3.4123,1) # rounding mathematically
'%.1f' % -3.4123 # rounding just for display #1 - produces string, not number
'{0:.1f}'.format(-3.4123) # rounding just for display #2 - produces string, not number

#squaring
math.sqrt(344)  #module
344 ** .5       #expression
pow(344, .5)    #built-in

# the random module
import random
random.random()
random.randint(1, 100)

# choosing randomly and shuffling
movies = ['Life of Brian', 'Holy Grail', 'The Meaning of Life']
random.choice(movies)
random.shuffle(movies)

# the decimal type
# they are fixed-precision floating point values
#1. fixed decimal places
#2. custom rulos of rounding and truncation
3 * 0.1 - 0.3 # usual floating point numbers
from decimal import Decimal
Decimal('0.1') + Decimal('0.1') + Decimal('0.1') - Decimal('0.3')

import decimal
decimal.getcontext().prec = 2
decimal.Decimal.from_float(1.23456)

# changing precision temporarily with the with context manager statement

with decimal.localcontext() as ctx:
        ctx.prec = 1
        decimal.Decimal('1.33') / decimal.Decimal('2.54')


# fraction number type
# keeps numerator and denominator variables explicitly

from fractions import Fraction
x = Fraction(1, 4)
y = Fraction(2, 6)
x + y
print(y)
z = .25
Fraction(z)

# conversions
(2.5).as_integer_ratio()
f = 2.5
z = Fraction(*f.as_integer_ratio())
x + z
float(x + z)
Fraction.from_float(12.3)

# mixed types
z = Fraction(4, 7)
z + 2       # fraction + integer = fraction
z + 1.2     # fraction + float = float
# limit denominator to achieve accuracy when converting from floats
x = Fraction(1, 3)
a = x + Fraction(*(4.0 / 3).as_integer_ratio())
a.limit_denominator(10)

### sets
# iterable - shrinks and grows, but immutable
# no duplications
# mixed data types
# expression operations
# literal for sets is {}

x = set('scrubb')
y = set('shrubb')
x - y
x | y
x & y
x ^ y
x > y
's' in x    #membership test
# some methods for sets (they allow other data types)
z = x.intersection(y)
x.add('SPAM')
x.update(['x', 'y']) (# adding a list)
x.remove('r')

# using 3.X syntax
s = set('spamalot')
{1,2,3}.issubset(range(-5, 5))
{1, 2, 3}.union([5, 6]) # method allows mixed data types

# despite lists or dictionaries, tuple can be added to sets
s = set([1, 2, 3, 4, 5])
s.add((1, 2, 3))

# in order to nest a set within a set, the frozenset is needed

# set comprehensions
{x ** 2 for x in [1, 2, 3, 4, 5, 6]}
S = {x * 3 for x in 'spamelot'}
S | {'mmm', 'kkk'}

# sets for order neutral equality tests
L1 = list('spam')
L2 = list('pasm')
L1 == L2 # runs on error because in sequences order matters
set(L1) == set(L2)
sorted(L1) == sorted(L2)