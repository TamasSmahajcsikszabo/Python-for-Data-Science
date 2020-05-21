# __repr__ and __str__ methods

class adder:
    def __init__(self, value=0):
        self.data = value
    def __add__(self, other):
        self.data += other
    def __repr__(self):
        return 'addrepr(%s)' % self.data
    def __str__(self):
        return '[Value: %s]' % self.data
x = adder(8)
print(x)
bin(5)
hex(8)
help(range)
