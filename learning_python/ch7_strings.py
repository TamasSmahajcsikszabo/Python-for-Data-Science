ű### strings are immutable sequences, left-to-right order
import re
# re is the regular expression module

# basic literals
s1 = 'shrubbery'
s2 = "shrubbery"
s1 == s2
s2 is s1

# implicit concatenation
'spam' ' and ' 'eggs'
# explicit concatenation
'spam' + ' and ' + 'eggs'

# escape sequences
s = 'a\nb\tc'
print(s)

# entering binary zero
'a\0b\0c'

# raw strings
# r or R at the beginning turns of the escape mechanism
path = r'c:\data\example'
print(path) # prints it without double \

# block strings (triple quoptes)
msg = """ Always look
at the bright
side of life! """
print(msg)
# triple quotes are good for disabling chunks of code

# 1. basic operations
str = 'foggy' + 'morning'
rep = 'repeat this' ' many times ' * 3
len(rep)

# loops and membership tests
job = 'hacker'
for c in job: print(c, end = ' ')
'k' in job

# slicing and offsetting
str = 'camelot'
str[0:3]
str[-1] # minus offset is added to the total length 
str[:-2] # left offset is inclusive, right offset is noninclusive
str[2:]

# extended slicing (step, a.k.a. stride)
s = 'asbfbviernoemrtgmprpotmgprm'
s[0::3]
s[::-1] # negative order
# slice objects
s[slice(0,3)]
s[slice(None, None, -1)]

### chr and ord
# chr converts an ASCII code
chr(1112)
# ord prints a character's ASCII code
ord('a')

B = '1101'
I = 0
while B != '':
    I = I * 2 +(ord(B[0]) - ord('0'))
    B = B[1:]

I

# changing strings
S = 'spam'
S = S + 'Burger' + '!'

S = 'splot'
S = S.replace('pl', 'pamal')

# creating new strings by formatting
'That is %d %s bird!' % (1, 'dead')         # format expression
'That is {0} {1} bird!'.format(1, 'dead')   # format method

### string methods
# only one replace
s = 'aass$$aas$$asasa$$asasa$$'
s = s.replace('$$', 'SPAM')

where = s.find('SPAM')
s = s[:where] + 'EGGS' + s[(where+4):]

s = 'aass$$aas$$asasa$$asasa$$'
s = s.replace('SPAM', 'EGGS', 1)

# splitting into mutable objects
s = 'spam'
s = list(s)
s = ''.join(s)      # converting back
s2= ' '.join(s)

### parsing text
line = 'aaa bbb ccc'
cols = line.split()     # the delimiter defaults to whitespace

line2 = 'aaa_bbb_ccc'
cols2 = line2.split('_')

### more advanced tools
import csv
help(csv)

# stripping, case setting and other operations
line = 'The kinghts who say Ni!\n'
line.rstrip()
line.upper()
line.isalpha()
line.endswith('Ni!\n')
line.startswith('The')

### string formatting expressions or method calls
# 1. expressions
exclamation = "Ni"
'The knights who say %s!' % exclamation

x = 1234
res = 'integers: ...%d...%-6d...%06d' % (x,x,x)

x = 2.223456789
'%e | %f | %g' % (x,x,x)

'%1.4f' % (x)
'%.*f' % (4, x)         # using * operator

# dictionary based formatting
'%(qty)d more %(food)s' % {'qty' : 1, 'food' : 'spam'}

reply = '''
Greetings
Hello %(name)s!
Your age is %(age)s
'''
values = {'name' : 'Bob', 'age' : 44}
print(reply % values)

# using vars()
food = 'spam'
qty = 10
'%(qty)d more %(food)s' % vars()

# 2. methods

# by position
template = '{0}, {1} and {2}'
template.format('ham', 'egg','Gumby')

# by keyword
template = '{motto}, {food} and {pork}'
template.format(motto = 'spam', food = 'eggs', pork = 'ham')


# by relative position
template = '{}, {} and {}'
template.format('ham', 'egg','Gumby')

import sys
'My {1[kind]} runs {0.platform}'.format(sys, {'kind' : 'laptop'})
sys.platform

'My {map[kind]} runs {sys.platform}'.format(sys = sys, map = {'kind' : 'laptop'})

somelist = list('spam')
'first = {0[0]}, third = {0[2]}'.format(somelist)
'first = {0}, last = {1}'.format(somelist[0], somelist[-1])
parts = somelist[0], somelist[-1], somelist[1:3]
'first = {0}, last = {1}, middle = {2}'.format(*parts) # *parts unpacks a tuple's items into individual arguments

### advanced string formatting syntax with methods
'{0:>6} = {1:<10}'.format('spam', 1234.123) # left and right alignment
'{1:12}'.format()

# thousand separator in use
'{0:,.2f}'.format(2145.578)     # note the ',' before the .2f part

# exercises
l = [1,2,3,4]
l.find("2")

s= "s,pa,m"
s.split(',')[1]
s[2:4]