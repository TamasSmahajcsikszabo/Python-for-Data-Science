# __getitem__
# it is called for instance-indexing operations

from squares import Squares

class Indexer:
    def __getitem__(self, index):
        return index**2


X = Indexer()
X[3]

for i in range(5):
    print(X[i], end=', ')

# it is also called for slicing
# for slicing a slice object is used
L = [5, 6, 7, 8, 9]
L[slice(1, 4)]


class Indexer:
    data = [4, 5, 6, 7, 8, 9]

    def __getitem__(self, index):
        print('getitem:', index)
        return self.data[index]


X = Indexer()
X[3]
X[1:3]  # when called for slicing, the method receives a slice object

# getting the getitem() method slice bounds


class Indexer:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        if isinstance(index, int):
            print('indexing at', index, ':', self.data[index])
        else:
            print('slicing between', index.start, index.stop, ':',
                  self.data[index])


X = Indexer([4, 5, 6, 7])
X[3]
X[1:5:2]

# the setitem() method


class Indexer:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        if isinstance(index, int):
            print('indexing at', index, ':', self.data[index])
        else:
            print('slicing between', index.start, index.stop, ':',
                  self.data[index])

    def __setitem__(self, index, value):
        self.data[index] = value
        print(self.data)


X = Indexer([4, 5, 6, 7, 8, 9])
X[1] = 3

# __index__ method:


class C:
    def __index__(self):
        return 5


X = C()
hex(X)


# for loops also call thr __getitem__ method
class stepper:
    def __getitem__(self, i):
        return self.data[i]


c = stepper()
c.data = "the knights of ni"

for ch in c:
    print(ch)

# a class that responds to for loops, also works in all iteration contexts, such as:
'o' in c  # membership tests
[ch for ch in c]  # list comprehensions
list(map(str.upper, c))  # map calls
c = stepper()
c.data = 'tre'
(h, a, m) = c  # sequence assignment

# __iter__
# supports more general iteration contexts than getitem()
# when no iteration protocol defined, Python falls back to getitem and does repeated indexing

for i in Squares(2, 5):
    print(i, end=' ')

sq = Squares(1, 4)
iter(sq)
next(sq)

# squares is in this form a one-shot iterator, turning into a list save this, but at a performance cost
X = Squares(1, 5)
tuple(X), tuple(X)  # by the second call, it's already empty

X = list(Squares(1, 5))
# by the second call, it's still the same list, can be iterated again
tuple(X), tuple(X)

# the same solution with generator functions or expressions
# NOTE
# they automatically produce iterable objects and retain local vaiable state between iterations


def sqr(start, stop):
    for i in range(start, stop + 1):
        yield i**2


list(sqr(1, 3))

for i in (i**2 for i in range(1, 4)):
    print(i, end=' ')

# single vs. multiple scans with classes
# to achieve multiple class iter() should not  return self
# rather it needs to define a new stateful object for the iterator


class SkipObject:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __iter__(self):
        return SkipIterator(self.wrapped)


class SkipIterator():
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.offset = 0

    def __next__(self):
        if self.offset >= len(self.wrapped):
            raise StopIteration
        else:
            item = self.wrapped[self.offset]
            self.offset += 2
            return item


alpha = 'abcdef'
skipper = SkipObject(alpha)
I = iter(skipper)
print(next(I), next(I), next(I))

# test multiple scans of objects created as classes
for x in skipper:
    for y in skipper:
        print(x + y, end=' ')

# iter() with yield generator function
# it supports multiple iterations


class Squares:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __iter__(self):
        for value in range(self.start, self.stop + 1):
            yield value**2


S = Squares(1, 5)
I = iter(S)
iter(I), iter(I), iter(I)
next(I), next(I), next(I)

# to reproduce multiple scan iteration without yield:


class Squares:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __iter__(self):
        return SquaresIter(self.start, self.stop)


class SquaresIter:
    def __init__(self, start, stop):
        self.value = start - 1
        self.stop = stop

    def __next__(self):
