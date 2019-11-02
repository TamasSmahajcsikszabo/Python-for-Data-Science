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
