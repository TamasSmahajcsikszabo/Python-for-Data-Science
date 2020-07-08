import array

# create array from a list
import numpy as np
np.array(list(range(1,10)))
A = np.array([1,2,3,4,5], dtype = 'float32')
np.array([range([i, i+3])] for i in [2,4,6])