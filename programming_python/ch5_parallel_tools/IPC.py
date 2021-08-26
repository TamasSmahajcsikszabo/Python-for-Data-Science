# inter process communication

# anonymous pipes
# ~ undirectional channel with standard Python file call
# ~ FIFO scheme, it allows synchronizing file execution
# ~ anonymous (they live within processes, link parent and child in forks) or
# ~ names (fifos, i.e. named files)

import threading
import time
import os
import time
import _thread
stdoutlock = _thread.allocate_lock()


def child(pipeout):
    zzz = 0
    while True:
        time.sleep(zzz)
        childid = os.getpid()
        msg = (
            'Spam %03d from %d' %
            (zzz, childid)).encode()  # pipes are binary pipes
        os.write(pipeout, msg)
        zzz = (zzz+1) % 5


def parent():
    # returns a tuple of two file descriptors: low-level file identifiers for
    # input and output of a pipe
    pipein, pipeout = os.pipe()
    if os.fork() == 0:
        child(pipeout)
    else:
        while True:
            with stdoutlock:
                line = os.read(pipein, 32)
                print(
                    'Parent %d got [%s] at %s' %
                    (os.getpid(), line, time.time()))
                if input() == 'q':
                    break


parent()


# wrapping pipe descriptors in file objects:
# closing unused pipe ends


def child(pipeout):
    zzz = 0
    while True:
        time.sleep(zzz)
        childid = os.getpid()
        msg = (
            'Spam %03d from %d\n' %
            (zzz, childid)).encode()  # note end of line!
        os.write(pipeout, msg)
        zzz = (zzz+1) % 5


def parent():
    # returns a tuple of two file descriptors: low-level file identifiers for
    # input and output of a pipe
    pipein, pipeout = os.pipe()
    if os.fork() == 0:
        os.close(pipein)
        child(pipeout)
    else:
        os.close(pipeout)
        pipein = os.fdopen(pipein)
        while True:
            line = pipein.readline()[:-1]
            print(
                'Parent %d got [%s] at %s' %
                (os.getpid(), line, time.time()))


parent()


# anonymous pipes between threads

def child(pipeout):
    zzz = 0
    while True:
        time.sleep(zzz)
        childid = os.getpid()
        msg = (
            'Spam %03d from %d' %
            (zzz, childid)).encode()  # note end of line!
        os.write(pipeout, msg)
        zzz = (zzz+1) % 5

def parent(pipein):
    while True:
        line = os.read(pipein, 32)
        print(
            'Parent %d got [%s] at %s' %
            (os.getpid(), line, time.time()))

pipein, pipeout = os.pipe()
threading.Thread(target = child, args=(pipeout, )).start()
parent(pipein)


