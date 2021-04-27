# zip, enumerata and comprehensions
ratios = [0.1, 0.6, 0.3]
colors = ['red', 'blue', 'green']


def get_color_ratios(colors, ratios):
    assert len(colors) == len(ratios)
    return dict(zip(colors, ratios))

get_color_ratios(colors, ratios)


# comparing dates
class
