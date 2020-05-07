# broadcasting
#applying element by element binary unfuncs on arrays and scalars and different sized variables
import numpy as np
a = 5
b = np.array([1,2,3])
a + b # a is mentally extended to match the longer array in the operation, in reality no duplication occurs
# in general, smaller element broadcast or stretch across larger objects

c = np.ones([3,3])
b + c

#both variables are streched to match a common shape
a = np.array([1,2,3])
b = a[:, np.newaxis]
a + b

M = np.ones((2,3))
a = np.arange(3)
M + a

# practical use cases
# centering an array
X = np.random.random((10, 3))
Xmean = X.mean(axis = 0) # broadcasting
Xcentered = X - Xmean

# 2D functions
x = np.linspace(0,5, 50)
y = np.linspace(0,5,50)[:, np.newaxis]

z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
import matplotlib.pyplot as plt

plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
plt.colorbar()
plt.savefig("Contour.png")
