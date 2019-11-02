import numpy as np
np.__version__

# fixed-type arrays in Python
import array as ar
L = list(range(10))
L = ar.array('i', L) #'i' is for indicating type

# NUmpy's ndarray objects
numpy_array = np.array(L)

# it only allows identical types
# automatic upcasting
np.array([3.14, 2,3,4])

# zero array
np.zeros(10, dtype = 'float32')

# shaped array filled with 1s
np.ones((3,5), dtype='int')

# custom shaped array
np.full((3,5), 3.14)

# range-like array
np.arange(0,20,2)

# array of values evenly paced
np.linspace(0,1,5)

# shaped array of uniformly distributed normal random values
np.random.random((3,3))

# shaped array of normally distributed random values
np.random.normal(1,3,(3,3))

# shaped array of min-max random values
np.random.randint(1,5,(3,3))

# 3x3 identity matrix
np.eye(3)
