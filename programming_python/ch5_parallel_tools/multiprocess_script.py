# Process is like Thread, runs parallel but lauches a process, not a thread

import os
from multiprocessing import Process, Lock


def whoami(label, lock):
    msg = '%s: name:%s, pid:%s'
    with lock:
        print(msg % (label, __name__, os.getpid()))


if __name__ == '__main__':
    lock = Lock()
    whoami('function call', lock)

    p = Process(target=whoami, args=('spawned child', lock))
    p.start()
    p.join()

    for i in range(5):
        Process(target=whoami, args=(('run process %d' % i), lock)).start()

    with lock:
        print('Main process exit.')

# other objects
# Pipe: bidirectional with two Connection objects at its ends
# Array/Value: shared process and thread-safe memory
# Queue: FIFO list of Python objects


