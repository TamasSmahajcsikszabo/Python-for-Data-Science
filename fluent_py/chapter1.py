import collections
from random import choice

Card = collections.namedtuple('Card', ['rank', 'suit'])
# we can use namedtuple to build classes of objectsthat are just bundles
# of attributes with no custom methods, like a database record


class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit)
                       for suit in self.suits for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

deck = FrenchDeck()
len(deck)
deck[5]
choice(deck) #no need for random picking method

suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)
def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

for card in sorted(deck, key=spades_high):
    print(card)
card = choice(deck)
spades_high(card)

# as it's immutable, FrenchDeck cannnot be reshuffled

# emulating numeric types
import math

class Vector:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Vector({0}, {1})'.format(self.x, self.y)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __radd__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

Vector(1,2) + Vector(0, 1)
Vector(2, 5) * 3
2 * Vector(3,4)


