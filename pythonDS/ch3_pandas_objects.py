import numpy as np
import pandas as pd

# series
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data.values  # produces similar to numpy array
data.index  # pandas Index object

data[1:3]
# numpy arrays have implicitly defined integer indices,
# pandas series have explicit indices associated with the values
# indices can be of any type, like strings

data = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# indices can be nonsquential, and noncontiguous

# pandas Series is an efficient Python dictionary
dictionary = {
    'name': 'Taylor',
    'profession': 'astronaut',
    'skillrating': 78,
    'year of disappearance': 1968
}
pandas_dict = pd.Series(dictionary)
pandas_dict['skillrating']  # keys have become indices

# unlike dictionaries, it supports slicing, too
pandas_dict['name':'skillrating']


# the DataFrame Object
# analog of two dimensional arrays with flexible row and column indices
# it's a sequence of aligned Series objects (aligned = sharing the same index)

area = {
    'Elvenland': 237323,
    'Deepdark': 445450,
    'Meadows': 23232
}

population = {
    'Elvenland': 23232,
    'Deepdark': 2200000,
    'Meadows': 23222
}
fantasy = pd.DataFrame({
    'population': population,
    'area': area
})

fantasy
fantasy.index
fantasy.columns

# keys are column references
fantasy['area']['Elvenland']

# constructing a DF:
# from a Series object
# from a list of dictionaries

data = [{'a': i, 'b': 2 * i} for i in range(4)]

pd.DataFrame(data)

# from a dictionary of a Series object
# from a 2-dimensional numpy array
pd.DataFrame(np.random.rand(3, 2), columns=['t1', 's2'])


# from a list
l = ['a', 'b', 'c']
pd.DataFrame({
    'test': l
})

# from a numpy structured array
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
pd.DataFrame(A)

# Index object
# it's like an immutable array or an ordered set
ind = pd.Index([1, 2, 3, 4, 5, 6])
ind[::-1]
ind.shape
ind.ndim
ind.dtype
ind.size
# unlike numpy arrays, it's immutable

# joins, unions, intersections
ind1 = pd.Index([1, 2, 3, 4, 5, 6, 7])
ind2 = pd.Index([5, 6, 7, 8, 9, 10])
ind1 & ind2  # intersection
ind1 | ind2  # union
ind1 ^ ind2  # symmetric difference

