from vector_drawing import draw, Points, Polygon
from math import sqrt
import typing
from numpy import pi

dino_vec = [
    (6, 4),
    (3, 1),
    (1, 2),
    (-1, 5),
    (-2, 5),
    (-3, 4),
    (-4, 4),
    (-5, 3),
    (-5, 2),
    (-2, 2),
    (-5, 1),
    (-4, 0),
    (-2, 1),
    (-1, 0),
    (0,-3),
    (-1,-4),
    (1,-4),
    (2,-3),
    (1,-2),
    (3,-1),
    (5,1)]
draw(Points(*dino_vec), Polygon(*dino_vec))


x = [(i, i**2) for i in range(-10,11)]
draw(Points(*x),grid=(1,10), nice_aspect_ratio=False)

class Vector:
    "Base class for vector"
    def __init__(self, x, y):
        self.x=x
        self.y=y
        self.length = sqrt(self.x**2+self.y**2)
        self.x_comp = (self.x, 0)
        self.y_comp =  (0, self.y)
        self.comps = [self.x_comp, self.y_comp]

    def __add__(self, _Vector):
        return Vector(x=self.x+_Vector.x, y=self.y+_Vector.y)

    def __sub__(self, _Vector):
        return self + (_Vector * (-1))

    def __repr__(self):
        return '{}, {}'.format(self.x, self.y)

    def as_tupple(self):
        return (self.x, self.y)

    def draw(self, color='red'):
        self.color=color
        return draw(Polygon(*[(0,0), (self.x, self.y)], color=self.color))

    def __mul__(self, other):
        return Vector(x=self.x*other, y=self.y*other)


a = Vector(1,5)
b = Vector(3,4)
b.length
b2=b*2
b * -1
b * (-1)
b2.length
c=a+b
b-a
c.length
Vector(-1, 2).length == (Vector(-1, 2) * (-1)).length
dino_Vec = [Vector(*dino_vec[i]).as_tupple() for i in range(len(dino_vec))]
draw(Points(*dino_Vec), Polygon(*dino_Vec))

new_dino = [Vector(-1.5, -2.5) + Vector(*dino_Vec[i]) for i in range(len(dino_Vec))]
new_dino = [new_dino[i].as_tupple() for i in range(len(new_dino))]
draw(Points(*new_dino), Polygon(*new_dino))

# Exercuse 2.6
u = Vector(-2, 0)
v = Vector(1.5, 1.5)
w = Vector(4, 1)

u + v
u + w
v + w
u + v + w
c= u+v
type(c)

# Exercise 2.7
def _add(*vectors):
    xs = sum([vectors[i][0] for i in range(len(vectors))])
    ys = sum([vectors[i][1] for i in range(len(vectors))])
    return (xs, ys)
_add([1,2],[2,4],[3,8],[4,16])


def _translate(translate: Vector, input = [Vector]) -> [Vector]:
    out = [(input[i] + translate).as_tupple() for i in range(len(input))]
    return out

e=_translate(translate=Vector(1,1), input=[Vector(0,0), Vector(0,1), Vector(-3, -3)])
typing.Tuple

input = [Vector(*dino_vec[i]) for i in range(len(dino_vec))]

def _clone(input: [Vector], ncol=10, nrow=10):
    # estimate size of the graph to make sure does not overlap:
    input_tuple = [input[i].as_tupple() for i in range(len(input))]
    x_range = [x[0] for x in input_tuple]
    x_range=(min(x_range), max(x_range))
    y_range = [x[1] for x in input_tuple]
    y_range=(min(y_range), max(y_range))
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    translations = []
    for i in range(nrow):
        for j in range(ncol):
            translations.append(Vector(x_span * (i+2), y_span*(j+2)))
    outobj=[_translate(trans, input) for trans in translations]

    out = [Polygon(*obj) for obj in outobj]
    draw(*out, grid=None, axes=None, origin=None)

_clone(input)


# 2.13
Vector(-6,-6).length
Vector(-6,-6).comps

Vector(5,-12).length
Vector(5,-12).comps


# 2.16
v = Vector(sqrt(2), sqrt(3))

w = v * pi
v.draw()
w.draw(color='blue')

def _scale(input: Vector, scalar:float) -> Vector:
    return input * scalar
_scale(Vector(2,2),2)

# 2.19
from random import uniform
u = Vector(-1,1)
v= Vector(1,1)

def generate_r():
    return uniform(-3,3)
def generate_s():
    return uniform(-1,1)

out= []
