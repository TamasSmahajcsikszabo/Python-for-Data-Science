import os
from multiprocessing import Process, Value, Array

procs = 3
count = 0


def showdata(label, val, arr):
    msg = '%-12s: pid:%4s, global:%s, value:%s, array:%s'
    print(msg % (label, os.getpid(), count, val.value, list(arr)))


def updater(val, arr):
    global count
    count += 1
    val.value += 1
    for i in range(3):
        arr[i] += 1


if __name__ == '__main__':
    scalar = Value('i', 0)  # integer
    vector = Array('d', procs)  # double

    # parent (main in this case) process values
    showdata('parent', scalar, vector)

    # create child and pass in shared memory
    p = Process(target=showdata, args=('child', scalar, vector))
    p.start(); p.join()

    # each process waits until the other finishes
    print('\nloop1 (updates in parent, serial children)...')
    for i in range(procs):
        count += 1
        scalar.value += 1
        vector[i] += 1
        p = Process(target=showdata, args=(('process %s' % i), scalar, vector))
        p.start()
        p.join()

    # allows parallel run
    print('\nloop2 (updates in parent, parallel children)...')
    ps = []
    for i in range(procs):
        count += 1
        scalar.value += 1
        vector[i] += 1
        p = Process(target=showdata, args=(('process %s' % i), scalar, vector))
        p.start()
        ps.append(p)
    for p in ps: p.join()

    # shared memory updated in children, wait for each
    print('\nloop2 (updates in serial children)...')
    for i in range(procs):
        p = Process(target=updater, args=(scalar, vector))
        p.start()
        p.join()
    showdata('parent temp', scalar, vector)

    # shared memory updated in children, allow parallel run
    print('\nloop2 (updates in parallel children)...')
    ps = []
    for i in range(procs):
        p = Process(target=updater, args=(scalar, vector))
        p.start()
        ps.append(p)
    for p in ps: p.join()

    showdata('final results: parent end', scalar, vector)
