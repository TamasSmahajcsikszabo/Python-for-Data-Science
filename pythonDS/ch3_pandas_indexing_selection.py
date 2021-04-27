import pandas as pd
import numpy as np

# Series
data = pd.Series([0.25, 0.5, 0.75, 1], index=['a', 'b', 'c', 'd'])
'a' in data
data.keys()
list(data.items())

# extension just like adding a new item to a dictionary
data['e'] = 17

# slicing
data['a':'c']  # inclusive
data[1:2]  # non-inclusive

# masking
data[(data > 0.3) & (data < 0.8)]

# fancy indexing
data[['a', 'c']]

# indexers: loc, iloc and ex
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data[1]  # explicit
data[1:3]  # implicit

# loc allows indexing and slicing, always on explicit
data.loc[1:3]
data.loc[1]

# iloc always uses implicit indexing
data.iloc[1]
data.iloc[1:3]
# principle: explicit is better than implicit


# data selection in dataframes
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

# indexing columns
fantasy['area']
fantasy.area  # works only if column names are not conflicting method names, or non-strings

# modifying with dictionary-style indexing
fantasy['density'] = fantasy['population'] / fantasy['area']
fantasy.values
fantasy.T  # transpose

# access rows
fantasy.values[0]
# access columns
fantasy['density']

# indexers
# iloc can be used to access the underlying data as numpy arrays
fantasy.iloc[:2, :2]  # row - column orderning
fantasy.loc[:'Deepdark', ]

fantasy.loc[fantasy.density > 1, ['area', 'population']]
fantasy.iloc[0, 1] = fantasy.iloc[0, 1] * 1.002

# masking and slicing are conducted row-wise
fantasy[1:2]
fantasy[fantasy.density < 4]
fantasy.loc[['Meadows', 'Elvenland'], ]
