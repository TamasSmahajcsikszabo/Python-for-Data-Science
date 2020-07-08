# namespace - each module is a self-contained namespace
# scope = nested namespace
# each module is a global scope, each function is a local
# function body assignments are associated only with the function's namespace

# lexical scoping - source code location determines the variables's place in namespace
#__main__ = the main interactive module
# the LEGB rule
# attributes follow a completely different lookup route
# special case is comprehension variable which is local for that expression
# try's exception variables are local to except block

X = 99

def func(Y):
    Z = X + Y
    return Z

func(1)
## local variables z and y are removed from the memory when the function call end, and
## the objects they reference may be garbage-collected if not referenced elsewhere

# the built-in scope - the last lookup place in LEGB
import builtins
dir(builtins)
# if overwritten a built-in, simple use: del name, which restores it
 
# global allows us to change global namespace objects
# in other words, it affects out of the def
# nonlocal does the same, but does not gp up until the level of the enclosing module, stays at the enclosing def

X = 1

def change():
    global X
    X += 1
    print(X)


[change() for i in list(range(1,6))]

#### GUIDELINE
# always use locals
# use as few globals as possible - i.e. don't let your functions be coupled to global objects
# two files can be coupled in the same way, don't do inplicit value changes between files

### nested scopes
x = 99

def f1():
    x = 88
    def f2():
        print(x) ### f2 references x which lives in the enclosing function's local scope
    f2()

f1()

# factory functions / closures

def maker(n):
    def action(x):
        return n ** x
    return action ## maker() makes function() without calling it

f = maker(2)
f ## it gets back with a reference to the generated nested function
f(3) # here we call the nested funcion maker created

# the nested function remembers the parameter 2 fed into the outer function
# by the time we feed 3 into the nested function, the outer maker() was already called and exited
# 2 is retained as state information attached to the generated action()

g = maker(3) # a new state information
g(3)

# this is common in landa function creation expressions
def maker(n):
    return lambda x: x ** n

t = maker(2)
t(3)

def func():
    x = 4
    action = (lambda n: x ** n)
    return action

y = func()
y(2)

# in nested loops def or lambda need defaults
def makeActions():
    acts = []
    for i in range(5):
        acts.append(lambda x, i=i: i ** x)
    return acts

acts = makeActions()
acts[0](2)
acts[1](2)
acts[2](2)

### NONLOCAL
# allows and limits scope lookups to the nested scope!
# it lets the local scope to be skipped from the lookup

def tester(start):
    state = start
    def nested(label):
        print(label, state)
    return nested

F = tester(0)
F('spam')

def tester(start):
    state = start
    def nested(label):
        nonlocal state
        print(label, state)
        state += 1
    return nested

F = tester(0)
F('spam')
F('ham')

G = tester(10)
G('spam')
G('ham')

### State Information Retention

#nonlocal
# 1. it creates a package of information
# state not visible outside of enclosing function

def tester(start):
    state = start
    def nested(label):
        nonlocal state
        print(label, state)
        state +=1
    return nested

f = tester(0)
f('spam')



# 2. state information with globals

def tester(start):
    global state # unlike nonlocal objects, globals does not need to be created before calling
    state = start
    def nested(label):
        global state
        print(label,state)
        state += 1
    return nested

f = tester(0)
f('spam')
state # because state is a global, it can naturally be referenced outside def
# leads to name collisions
# it allows a single shared copy of state - different function calls overwrite it

# 3. state information with classes
class tester:
    def __init__(self, start):
        self.state = start
    def nested(self, label):
        print(label, self.state)
        self.state += 1
f = tester(0)
f.nested('spam')
f.nested

class tester:
    def __init__(self, start):
        self.state = start
    def __call__(self, label):
        print(label, self.state)
        self.state += 1
f = tester(0)
f('spam')

# 4. function attributes
def tester(start):
    def nested(label):
        print(label, nested.state)
        nested.state += 1
    nested.state = start
    return nested

f = tester(0)
f('spam')