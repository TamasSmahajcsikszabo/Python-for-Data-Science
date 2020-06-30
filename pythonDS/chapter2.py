import array as ar
import numpy as np
np.__version__

# fixed-type arrays in Python
L = list(range(10))
L = ar.array('i', L)  # 'i' is for indicating type

# NUmpy's ndarray objects
numpy_array = np.array(L)

# it only allows identical types
# automatic upcasting
np.array([3.14, 2, 3, 4])
f32 = np.array([3.45, 2.34, 23.3], dtype='float32')

# multidimensional arrays initiated as nested lists
nested = np.array([range(i, i + 3) for i in [2, 4, 6]])

# generating arrays using built-in routines
# zero array
np.zeros(10, dtype='float32')

# shaped array filled with '1's
np.ones((3, 5), dtype='int')

# custom shaped array
np.full((3, 5), 3.14)

# range-like array
np.arange(0, 20, 2)

# array of values evenly paced
np.linspace(0, 1, 15)

# shaped array of uniformly distributed normal random values
np.random.random((3, 3))

# shaped array of normally distributed random values
np.random.normal(1, 3, (3, 3))

# shaped array of min-max random values
np.random.randint(1, 5, (3, 3))

# 3x3 identity matrix
np.eye(3)

# creating an uninitiated ineger array with length 5
np.empty(5)
