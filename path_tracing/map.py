def get_map(x ,y, body):
    map = ""

    for yi in range(y):
        for xi in range(x):
            test = xi == (x-1)
            if test:
                map = map + body + '\n'
            else:
                map = map + body
    return map


class Map():
    def __init__(self, width=50, height=25, body="."):
        self.width = width
        self.height = height
        self.body = body
        self.body_string = get_map(self.width, self.height, self.body)

    def __repr__(self):
        return self.body_string

map = Map()
