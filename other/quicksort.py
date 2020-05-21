import math
import numpy as np

def quicksort(input):
    # make sure the input is an array
    input = np.array(input)
    pivot = input[math.floor(len(input)/2)]
    print(pivot)
    smaller = np.array([])
    bigger = np.array([])

    for i in range(0, len(input) - 1):
        chosen_element = input[i]
        if (chosen_element != pivot):
            if (chosen_element < pivot):
                smaller = np.append(smaller, chosen_element)
            elif (chosen_element > pivot):
                bigger = np.append(bigger, chosen_element)
    smaller = np.append(smaller, pivot)
    for e in bigger:
        smaller = np.append(smaller, e)
    print(smaller)
    quicksort(smaller)

input = [3, 5, 7, 1, 9, 2, 22, 99, 11]
quicksort(input)
