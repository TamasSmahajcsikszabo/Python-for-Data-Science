## augmenting superclass method is subclass

class Super:
    def method(self):
        print('in Super.method')

class Sub(Super):
    def method(self):
        print('start sub.method')
        Super.method(self)
        print('end sub.method')

e = Sub()
e.method()

## see specialize.py for class interface techniques