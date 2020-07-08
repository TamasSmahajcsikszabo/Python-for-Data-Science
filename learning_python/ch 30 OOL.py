
## constructors and expressions
#__init__ and __sub__

class Number:
	def __init__(self, start):
		self.data = start
	def __sub__(self, other):
		return Number(self.data - other)


x = Number(5)
y = x - 4
print(y.data)

# indexing and slicing
# __getitem__ & __setitem__

class indexer:
	def __getitem__(self, index):
		return index ** 2
x = indexer()
x[2]

for i in range(5):
	print(x[i], end = ' => ')

class indexer:
	data = [5,6,7,8,9]
	def __getitem__(self, index):
		print('get item at:', index)
		return self.data[index]

x = indexer()
x[0]
x[1:2]
x[2:4]

class indexer:
	data = [5,6,7,8,9]
	def __getitem__(self, index):
		if isinstance(index, int):
			print('indexing at', index,':', self.data[index])
		else:
			print('slicing at', index.start, index.stop, index.step)

x = indexer()
x[4]
x[1:3]

# adding an indexing/slicing assignment method with __setitem__
class indexer:
	data = [5,6,7,8,9]
	def __getitem__(self, index):
		if isinstance(index, int):
			print('indexing at', index,':', self.data[index])
			return self.data[index]
		else:
			print('slicing at', index.start, index.stop, index.step)
			return self.data[index]
	def __setitem__(self, index, value):
		self.data[index] = value

x = indexer()
x[:] = list(range(1,110))
x.data
x[15]
x[1:43]

# __index__ is NOT for indexing
# a class with __getitem__ responds to for loops, too

class stepper:
	def __getitem__(self, index):
		return self.data[index]

x = stepper()
x.data = list(range(10, 65, 5))

x[3]
for i in x:
	print(i, end = " ")

# plus it also gets all other iteration tools:
import numpy as np
15 in x				  #membership
[(i+1)%5 for i in x]  # list comprehension
list(map(np.sqrt, x)) # map call
x.data = 'pame'
(a,b,c,d) = x		  #sequence assigment
a,b,c

list(x), tuple(x), ''.join(x)

# __iter__ method:
# it even support iteration protocol better than __getitem__

class Squares:
	def __init__(self, start, stop):
		self.value = start - 1
		self.stop = stop
	def __iter__(self):				# get the iterator object
		return self
	def __next__(self):
		if self.value == self.stop:
			raise StopIteration
		self.value += 1
		return self.value ** 2

for i in Squares(1,5):
	print(i, end = ' ')

#__iter__ with the usual iteration tools:
36 in Squares(1,10)
a,b,c = Squares(1,3)
[x for x in Squares(1,6)]
':'.join(list(map(str, Squares(1,10))))

# the same with generators:
def gsquares(start, stop):
	for i in range(start, stop+1):
		yield i**2

for i in gsquares(1,10):
	print(i, end=' ')

for i in (x**2 for x in range(1,11)):
	print(i, end=' ')

# supporting multiple iterations
# in the class, call a new stateful object instead of the self data
# a class example that supports multiple active loops directly
class SkipObject:
    def __init__(self, wrapped):
        self.wrapped = wrapped
    def __iter__(self):
        return SkipIterator(self.wrapped)

class SkipIterator:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.offset = 0
    def __next__(self):
        if self.offset > len(self.wrapped):
            raise StopIteration
        else:
            item = self.wrapped[self.offset]
            self.offset += 2
            return item

alpha = 'abcdef'
skipper = SkipObject(alpha)

I = iter(skipper)
print(next(I), next(I), next(I))

# independent iterators fetched from the same object, Squares couldn't do this
for x in skipper:
	for y in skipper:
		print(x+y, end = ' ')

# same results with slicing
x = "abcdef"

for x in x[::2]:
	for y in x[::2]:
		print(x + y, end = ' ')
# but this stores all slice results in memory


# a coding alternative
# __iter__ and yield

# start with a generator function
def gen(x):
	for i in range(x): yield i ** 2

G = gen(4)
list(G)

class squares:
	def __init__(self, start, stop):
		self.start = start
		self.stop = stop
	def __iter__(self):
		for value in range(self.start, self.stop + 1):
			yield value ** 2


for i in squares(1,5):
	print(i, end = ' ')

class squares:
	def __init__(self, start, stop):
		self.start = start
		self.stop = stop
	def gen(self):
		for value in range(self.start, self.stop + 1):
			yield value ** 2

s = squares(1,5)
list(s.gen())

# in the second variant gen() is a generator that returns itself
# if it's __iter__, then it return a new generator with __next__

# __iter__ with yield also support multiple active iterators
# each call to __iter__ is a call to a generator function that returns a new generator with its own copy of the
# local scope for state retention

S = squares(1,6)
I = iter(S)
J = iter(S)
next(I)
next(J)

for i in S:
	for j in S:
		print('%s <=> %s' % (i, j), end = ' ')

# the same without yield

class squares:
	def __init__(self, start, stop):
		self.start = start
		self.stop = stop
	def __iter__(self):
		return squaresIter(self.start, self.stop)

class squaresIter:
	def __init(self, start, stop):
		self.value = start -1
		self.stop = stop
	def __next__(self):
		if self.value == self.stop:
			raise StopIteration
		self.value += 1
		return self.value ** 2

S = squares(1,4)

for i in S:
	for j in S:
		print('%s - %s' % (i, j), end = ' ')

class skipper:
	def __init__ (self, wrapped):
		self.wrapped = wrapped
	def __iter__ (self):
		offset = 0
		while offset < len(self.wrapped):
			item = self.wrapped[offset]
			offset += 2
			yield item

s = skipper('abcdef')
for i in s:
	for j in s:
		print('%s:%s' % (i,j), end = ' ')

# Membership
# __contains__
# the following class contains all alternatives
class Iters:
	def __init__(self, value):
		self.data = value
	def __getitem__(self, i):
		print('get[%s]' % i, end = '')
		return self.data[i]
	def __iter__(self):
		print('iter=> ', end = '')
		self.ix = 0
		return self
	def __next__(self):
		print('next', end = '')
		if self.ix == len(self.data): raise StopIteration
		item = self.data[self.ix]
		self.ix += 1
		return item
	def __contains__(self,x):
		print('contains: ', end = '')
		return x in self.data

S = Iters([1,2,3,4,5])
print(3 in S)
for i in S:
	print(i, end = ' | ')

1 in S
S[1]

#multiple scans
class Iters:
	def __init__(self, value):
		self.data = value

	def __getitem__(self, i):
		print('get[%s]' % i, end='')
		return self.data[i]

	def __iter__(self):
		print('iter => next:', end='')
		for x in self.data:
			yield x
			print('next:', end='')

	def __contains__(self, item):
		print('contains:', end='')
		return item in self.data

a = Iters([1,2,3,4,5])
a[2]
for i in a: i

4 in a
a[1:]

# attribute references with the __getattr__ method
class empty:
	def __getattr__(self, attrname):
		if attrname == 'age':
			return 40
		else:
			raise AttributeError(attrname)

x = empty()
x.age # 40 as expected
x.length # exception

# __setattr__ calls another __setattr__
# to avoid loops, instead of self.name = x, use:
self.__dict__['name'] = x # assignment to attribute dictionary keys

class Accesscontrol:
	def __setattr__(self, attr, value):
		if attr == 'age':
			self.__dict__['age'] = value + 10
		else:
			raise AttributeError(attr + 'not allowed')

# __repr__ and __str__

class adder:
	def __init__(self, value):
		self.data = value
	def __add__(self, other):
		self.data += other

class stradder(adder):
	def __repr__(self):
		return 'addrepr(%s)' % self.data

x = stradder(2)
x + 1
x

# __str__ is for used-friendly display, for print
# __repr__ is for as.code string, detailed for developer

# while __repr__ is used generally as the fallback option for __str__, the reverse
# is not true

class stradder(adder):
	def __str__(self):
		return '[Value: %s]' % self.data

x = stradder(4)
x + 1
print(x)

# if both present, __str__ overrrides __repr__
class addboth(adder):
	def __str__(self):
		return '[Value: %s]' % self.data
	def __repr__(self):
		return 'addboth(%s)' % self.data

x = addboth(5)
x		 # __repr__
print(x) # __str__

class Printer:
	def __init__(self, value):
		self.data = value
	def __str__(self):
		return str(self.data)

objs = [Printer(2), Printer(3)]
for x in objs: print(x)
print(objs)
objs

# __radd__ & __iadd__
class Commuter:
	def __init__(self, value):
		self.data = value
	def __add__(self,other):
		print('add', self.data, other)
		return self.data + other
	def __radd__(self, other):
		print('radd', other, self.data)
		return other + self.data
x, y = Commuter(23), Commuter(43)

x + y
x + 1
y + 1
1 + x + 1 + y

# variants for calling __add__ in __radd__

# explicitly call __add__
class Commuter2:
	def __init__(self, value):
		self.data = value
	def __add__(self,other):
		print('add', self.data, other)
		return self.data + other
	def __radd__(self, other):
		return self.__add__(other)

x, y = Commuter2(23), Commuter2(43)

x + y
x + 1
y + 1
1 + x + 1 + y

# swap order
class Commuter3:
	def __init__(self, value):
		self.data = value
	def __add__(self,other):
		print('add', self.data, other)
		return self.data + other
	def __radd__(self, other):
		return self + other

x, y = Commuter3(23), Commuter3(43)

x + y
x + 1
y + 1
1 + x + 1 + y

# alias for __radd__
class Commuter4:
	def __init__(self, value):
		self.data = value
	def __add__(self,other):
		print('add', self.data, other)
		return self.data + other
	__radd__ = __add__
x, y = Commuter4(23), Commuter4(43)

x + y
x + 1
y + 1
1 + x + 1 + y

# propagating
class Commuter5:
	def __init__(self, value):
		self.data = value
	def __add__(self,other):
		if isinstance(other, Commuter5):
			other = other.data
		return Commuter5(self.data + other)
x, y = Commuter5(23), Commuter5(43)

x + y
x + 1
y + 1
1 + x + 1 + y
