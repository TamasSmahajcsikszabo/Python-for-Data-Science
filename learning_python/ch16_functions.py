# def is also a compund statement
# can be nested inside if, for and while and even another def
# their inner code is not run until the function is called!
# can have attributes

def times(x,y):
    return x * y
times(5, 6)
## functions are typeless:
times('Ni! ', 10).rstrip()
# this type-dependent behaviour is polymorphism
times(['a'], 4)

#intersecting sequences:
def intersect(seq1, seq2):
    catch = []                      ###local variable which exists only while the function runs, and disapears as the run is over
    for x in seq1:
        if x in seq2:
            catch.append(x)
    return catch
x = 'spam'
y = 'scam'
intersect(x, y)

s1 = 'spam'
s2 = 'scam'
[x for x in s1 if x in s2]
intersect([1,2,4], (1,4)) # mixed data types as inputs

# interface =  a set of methods and expression operators the functions's code runs