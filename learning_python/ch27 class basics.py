class FirstClass:
    def setnames(self, value):
        self.data = value
    def display(self):
        print(self.data)

x = FirstClass()
y = FirstClass()

x.setnames("King Arthur")
y.setnames(3.1243)

x.display()
y.display()

class Secondclass (FirstClass):
    def display(self):
        print('Current value = "%s"' % self.data)

z = Secondclass()
z.setnames(321)
z.display()


class ThirdClass(Secondclass):
    def __init__(self, value):
        self.data = value
    def __add__(self, other):
        return ThirdClass(self.data + other)
    def __str__(self):
        return '[ThirdClass: %s]' % self.data
    def mul(self, other):
        self.data *= other

a = ThirdClass('abc')
a.display()
print(a)

b = a + 'xyz'
b.display()
a.mul(3)
a.display()

class Person:
    def __init__(self, name, jobs, age=None):
        self.name = name,
        self.jobs = jobs,
        self.agee = age
    def info(self):
        return(self.name, self.jobs)

p1 = Person('Joe', ['mgr','arch'], 40)
p2 = Person('Anne', ['tchr'], 23)

p2.info()