import numpy as np
np.random.seed(0)

x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3, 2))
x3 = np.random.randint(10, size=(2, 3, 5))

# basic attributes
print("ndim: ", x1.ndim)
print("shape: ", x1.shape)
print("size: ", x1.size)


def get_attr(ids):
    """ gets numpy array attributes"""
    for i in range(0, len(ids)):
        obj = ids[i]
        varname = "x"
        varname = varname + str(i)
        print("[[ variable:", varname, "]]")
        print("ndim:", obj.ndim)
        print("shape:", obj.shape)
        print("size:", obj.size)
        print("dtypte:", obj.dtype)
        print("itemsize:", obj.itemsize, "bytes")
        print("nbytes:", obj.nbytes, "bytes")
        print("\n")


get_attr([x1, x2, x3])

# slicing and inserting, automatic truncation
a = np.array([2, 3, 4, 5, 6])
a[0] = 3.24

# slicing subarrays x[start:stop:step]
a = np.arange(20)
a[:10]
a[::2]
a[::-1]
a[12::-3]

multi = np.random.randint(10, size=(4, 4, 3))

# first row of every layer
multi[:, :1]

# all reversed
multi[::-1, ::-1, ::-1]

# first row of first layer
multi[0, 0, :]

# first column of second layer
multi[1, :, 1]

# by default subarrays are no-copy views, but they can be copied with the copy() method
array = np.random.randint(10, size=(5, 5))
array[:3, :2]
array[2, 0] = 1
array_sub = array[:3, :2].copy()

# reshaping
array = np.arange(1, 11)
array.reshape(5, 2)
array[np.newaxis, 5:]
array[:, np.newaxis]

# concatenation
np.random.seed(12)
a1 = np.random.randint(10, size=(2, 2))
a2 = np.random.randint(10, size=(2, 2))
np.concatenate((a1, a2), axis=1)
np.concatenate((a1, a2), axis=0)

# for arrays with different dimensions, stacking is advised
b1 = np.array([1, 2, 3])
b2 = np.array([[3, 4, 5], [6, 7, 8]])

np.vstack([b1, b2])
b3 = np.array([[88], [88]])
np.hstack([b2, b3])

# dstack concatenates along the third axis
np.dstack([b2, b2])

# splitting arrays
array = np.arange(16).reshape(4, 4)

np.split(array, [1, 3])
left, right = np.hsplit(array, [2])
print(left)

upper, lower = np.vsplit(array, [2])
print(lower)
