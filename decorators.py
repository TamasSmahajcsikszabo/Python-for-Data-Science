def formatter(func):
    res = func()
    return 'Results are: {}'.format(res)

@formatter
def pi():
    return 3.14

pi()
