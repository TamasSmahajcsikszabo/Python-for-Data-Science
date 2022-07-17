from vector_drawing import draw, Points, Polygon

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
    def __init__(self, x, y):
        self.x=x
        self.y=y

    def __add__(self, _Vector):
        return Vector(x=self.x+_Vector.x, y=self.y+_Vector.y)

    def __repr__(self):
        return '{}, {}'.format(self.x, self.y)


a = Vector(1,-1)
b = Vector(3,4)
c=a+b
c
