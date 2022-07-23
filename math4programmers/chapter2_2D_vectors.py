from vector_drawing import draw, Points, Polygon
from math import sqrt

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

    def __add__(self, _Vector):
        return Vector(x=self.x+_Vector.x, y=self.y+_Vector.y)

    def __sub__(self, _Vector):
        return self + (_Vector * (-1))

    def __repr__(self):
        return '{}, {}'.format(self.x, self.y)

    def as_tupple(self):
        return (self.x, self.y)

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

# Exercise 2.7
def _add(*vectors):
    xs = sum([vectors[i][0] for i in range(len(vectors))])
    ys = sum([vectors[i][1] for i in range(len(vectors))])
    return (xs, ys)
_add([1,2],[2,4],[3,8],[4,16])
