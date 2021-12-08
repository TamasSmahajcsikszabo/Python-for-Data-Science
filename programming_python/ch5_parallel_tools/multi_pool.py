import os
import sys
import time
from multiprocessing import Pool


def powers(x):
    return 2 ** x

if __name__ == '__main__':
    start = time.perf_counter()
    workers = Pool(processes=int(sys.argv[1]))
    size = int(sys.argv[2])

    results = workers.map(powers, [2]*size)

    results = workers.map(powers, range(size))
    end = time.perf_counter()

    print('computation ran in %d' % round(end - start, 2), "s")
