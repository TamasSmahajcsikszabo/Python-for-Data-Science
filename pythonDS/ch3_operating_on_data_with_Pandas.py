import pandas as pd
import numpy as np

# pandas preserves index and column labels for unary functions
# and for binary operators, it aligns indices


rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
df = pd.DataFrame(rng.randint(0, 10, (3, 4)), columns=['A', 'B', 'C', 'D'])

# any Numpy ufuncs will preserve indices of these objects
np.exp(ser)
np.sin(df * np.pi / 4)

