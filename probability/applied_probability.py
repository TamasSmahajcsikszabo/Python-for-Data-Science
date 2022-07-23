import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import scipy
from scipy import stats
from collections import Counter
import random


# basic definitions
# random = not predictable

# kolmogorov:
# 1. p >= 0 of 1 event, real number
# 2. sum p = 1
# 3. p of mutually exclusive events = sum p_1..k

# sequence of events
# 2 dice: 2/36 -> # of events, total # of possible events
# PDF: p of all possible outcomes
# cumulative PDF: p of outcome smaller or equal to each outcome P(X<x)

# PD with coins

b = stats.binom(100, 0.3)
# binomial - centered on n * p

class Coin:
    def __init__(self):
        self.sides = ['head', 'tail']

    def flip(self, n=10):
        self.flips_num = [random.randint(0,1) for i in range(n)]
        self.flips= [self.sides[i] for i in self.flips_num]
        return self.flips

test = Coin()
test.flip()
test.flips_num
