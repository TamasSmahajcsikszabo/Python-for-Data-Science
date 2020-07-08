
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
	def __inint__(self, start, stop):
		self.start = start,
		self.stop = stop
	def __iter__(self):
		for value in range(self.start, self.stop + 1):
			yield value ** 2


for i in square(1,5):
	print(i, end = ' ')