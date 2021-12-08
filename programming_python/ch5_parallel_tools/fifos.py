# named pipes / fifos
# independent named files not relying on shared memory
# can be used by threads, processes or programs too
# they are unidirectional
# two give a bidirectional setting
# fifo access is synchronized by OS

import os
import time
import sys

fifoname = '/tmp/fifopipe'


def child():
    zzz = 0
    pipeout = os.open(fifoname, os.O_WRONLY)
    while True:
        time.sleep(zzz)
        childid = os.getpid()
        msg = (
            'Spam %03d from %d' %
            (zzz, childid)).encode()  # pipes are binary pipes
        os.write(pipeout, msg)
        zzz = (zzz+1) % 5


def parent():
    pipein = os.open(fifoname, 'r')
    while True:
        line = pipein.readline()[:-1] # skipping '\n'
        print(
            'Parent %d got [%s] at %s' %
            (os.getpid(), line, time.time()))



if __name__ == '__main__':
    if not os.path.exists(fifoname):
        os.mkfifo(fifoname)
    if len(sys.argv) == 1: # i.e. no args provided
        parent()
    else:
        child()

