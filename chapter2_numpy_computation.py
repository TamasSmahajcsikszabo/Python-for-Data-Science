import numpy as np
np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output


values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)

big_array = np.random.randint(1, 10, size=1000000)
%timeit compute_reciprocals(big_array)

# NumPy ufuncs
%timeit print(1.0 / big_array)

theta = np.linspace(0, np.pi, 3)
print("theta:", theta)
print("sin(theta):", np.sin(theta))
print("cos(theta):", np.cos(theta))
print("tan(theta):", np.tan(theta))

# exponentials
x = np.linspace(1, 8, num = 1000)
np.power(x, 3)

# logarithms
np.log(x)
np.log2(x)
np.log10(x)

from scipy import special as s
s.gamma(x)
s.gammaln(x)
s.erf(x)

# use of the 'out' parameter
# writing output of calulcations into memory location
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out = y)
z = np.zeros(10)
np.power(2, x, out=z[::2])
print(z)

# aggregates
# reduce
np.add.reduce(x)
np.add.accumulate(x) # stores intermediate steps (results)

#summary functions
L = np.random.random(100)
sum(L)
np.sum(L)

big_array = np.random.random(1000000)
%timeit sum(big_array)
%timeit np.sum(big_array)

%timeit min(big_array)
%timeit np.min(big_array)

#the same we can compute max functions

# Operations alongside axis
d = np.random.random((3, 4))
np.max(d, axis=0) #rowwise
np.max(d, axis=1) #columnwise

# case study
import pandas as pd
data = pd.read_csv('~/repos/PythonDataScienceHandbook/notebooks/data/president_heights.csv')
heights = np.array(data['height(cm)'])
print('Mean height', np.mean(heights))
print('st. dev. of height', np.std(heights))
print('Min height', np.min(heights))
print('Max height', np.max(heights))
print('25th percentile height', np.percentile(heights, 25))
print('Median height', np.median(heights))
print('75th percentile height', np.percentile(heights, 75))

import matplotlib.pyplot as plt
import seaborn; seaborn.set()

plt.hist(heights)
plt.show()
