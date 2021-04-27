# decorator =  a callable that returns another callable

# automactically run code at the end of funcion and class definition statements
# augmenting function calls
# installs wrapper (a.k.a. proxy) objects to be invoked later:
# 1. call proxies: function call interception and managing
# 2. interface proxies: class decorators intercept late class creation
# decorators achieve these by automatically rebind function and class names to
# callables at the end of def and class calls
# these callables perform tasks such as tracing and timing, etc.
# other uses of decorators:
#   > function managing, managing function objects; e.g. object registration to
#   and API
#   > managing classes and not just instance creation calls, such as adding new
#   methods to classes
# they can be used to manage function/class creation/calls/instances and
# function/class objects, too

# function decorators
# runs one function through another at the end of a def statement, and rebinds
# the original function name to the results
# it's a runtime declaration
# @ + a reference to a metafunction
from decorator import decorator

def func():
    t = "Hello!\n"
    print(t)
    return True

import pdb; pdb.set_trace()
func = decorator(func) # rebind function name to decorator result

# a decorator is a callable that returns a callablel it returns the object to
# be called later when the decorated function is invoked through its original
# name;
# the decorated function is still available in an enclosing scope; the same
# applies for classes, the original decorated function is still available in
# an instance attribute, for example

class decorator(object):
    def __init__(self, func):
        self.func = func        # retains state
    def __call__(self, *args):


# supporting method decoration


