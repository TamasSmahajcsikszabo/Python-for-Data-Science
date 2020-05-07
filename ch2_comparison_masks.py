# Boolean masking for manipulating arrays
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

rainfall = pd.read_csv('../PythonDataScienceHandbook/notebooks/data/Seattle2014.csv')['PRCP'].values
inches = rainfall / 254
inches.shape

plt.hist(inches)
plt.savefig("Histogram of Seattle rainfall.png")

# comparison operations result in numpy arrays with Boolean data types 
rng = np.random.RandomState(0)
x = rng.randint(10, size=(3,4))
x < 6
np.count_nonzero(x < 6) == np.sum(x < 6)
np.sum(x < 6, axis = 1)
np.sum(x < 6, axis = 0)
np.all(x < 5)
np.any(x < 10, axis = 1)

print("Number of days without rain:     ", np.sum(inches == 0))
print("Number of rainy days:            ", np.sum(inches != 0))
print("Rainy days with more than 0.5 in:", np.sum(inches > 0.5))
print("Rainy days with < 0.1 in:        ", np.sum((inches > 0) & (inches < 0.2)))

rainy = (inches > 0)
summer = (np.arange(365) - 172 < 90) & (np.arange(365) - 172 > 0)

print("Median precip on rainy days in 2014 (inches):    ", np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches):    ", np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches):    ", np.median(inches[summer]))
print("Median precip on non-summer days in 2014 (inches):    ", np.median(inches[rainy & ~summer]))
