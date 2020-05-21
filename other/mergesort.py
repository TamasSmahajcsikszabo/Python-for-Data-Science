import numpy as np
import math

def ascending(data):
    result = []
    for i in range(0, len(data)):
        if i != (len(data) -1):
            result.append(data[i] < data[i + 1])
        else:
            result.append(data[i] > data[i-1])

    return all(result)

def sort(data):
    result = np.array([])
    pivot = data[int(np.random.randint(low = 0, high = len(data)-1, size = 1))]
    for i in data:
        if (i > pivot):
            result = np.append(result, i)
        else:
            result = np.append(i, result)
    print(result)
    return result
    while (ascending(data)):
        sort(result)

def mergesort(data):
    data = np.array(data)
    pivot = data[math.floor(len(data)/2)]
    
    subarray1 = data[0:math.floor(len(data)/2)]
    subarray2 = data[math.floor(len(data)/2):len(data)]
    
    for s in (subarray1, subarray2):
        s = sort(s)
        mergesort(s)



data = [45,66,88,22,11,55,99]
data1 = [1,2,3,4,5]
mergesort(data)
sort(data)
sorted = sort(data)
ascending(sorted)

