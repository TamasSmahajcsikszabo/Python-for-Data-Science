# __getitem__
# it is called for instance-indexing operations 

class Indexer:
    def __getitem__(self, index):
        return index ** 2

X = Indexer()
X[3]

for i in range(5):
    print(X[i], end=', ')

# it is also called for slicing
# for slicing a slice object is used
L = [5, 6, 7, 8, 9]
L[slice(1,4)]

class Indexer:
    data = [4, 5, 6, 7, 8, 9]
    def __getitem__(self, index):
        print('getitem:', index)
        return self.data[index]

X = Indexer()
X[3]
X[1:3] #when called for slicing, the method receives a slice object

# getting the getitem() method slice bounds
class Indexer:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        if isinstance(index, int):
            print('indexing at',index,':', self.data[index])
        else:
            print('slicing between', index.start, index.stop, ':', self.data[index])

X = Indexer([4,5,6,7])
X[3]
X[1:5:2]

# the setitem() method
class Indexer:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        if isinstance(index, int):
            print('indexing at',index,':', self.data[index])
        else:
            print('slicing between', index.start, index.stop, ':', self.data[index])
    def __setitem__(self, index, value):
        self.data[index] = value
        print(self.data)

X = Indexer([4,5,6,7,8,9])
X[1] = 3

# __index__ method:
class C:
    def __index__(self):
        return 5
X = C()
hex(X)
