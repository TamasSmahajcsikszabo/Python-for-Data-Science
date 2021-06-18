# fork processes
import _thread
import os
import time


def child():
    print('Hello from child', os.getpid())
    os._exit(0)


def parent():
    while True:
        newpid = os.fork()
        if newpid == 0:
            child()
        else:
            print('Hello from parent', os.getpid(), newpid)
        if input() == 'q':
            break


parent()


def counter(count):
    for i in range(count):
        time.sleep(1)
        print('[%s] => %s' % (os.getpid(), i))


for i in range(5):
    pid = os.fork()
    if pid != 0:
        print('Process %d spawned' % pid)
    else:
        counter(5)
        os._exit(0)
print('Main process exiting.')


# threads


def child(pid):
    print('Hello from child', pid)


def parent():
    i = 0
    while True:
        i += 1
        _thread.start_new_thread(child, (i,))
        if input() == 'q':
            break


parent()


def counter(myId, count):
    for i in range(count):
        time.sleep(1)
        print('[%s] => %s' % (myId, i))


for i in range(5):
    _thread.start_new_thread(counter, (i, 5))

time.sleep(6)
print('Main thread exiting.')

# namespaces and objects are shared among all threads
# one object change can be seen by all, main or child alike
# two threads changing one object at the same time might lead to object
# corruption
# _thread achieves this with *locks*

import _thread, time

def counter(myId, count):
    for i in range(count):
        time.sleep(1)
        mutex.acquire()
        print('[%s] => %s' % (myId, i))
        mutex.release()

# the lock ensures mutually exclusive access to the stdout stream
# output never overlaps as two threads never execute print calls at the same
# point in time
mutex = _thread.allocate_lock()

for i in range(5):
    _thread.start_new_thread(counter, (i, 5))

time.sleep(6)
print('Main thread exiting')


stdoutmutex = _thread.allocate_lock()
exitmutexes = [_thread.allocate_lock() for i in range(10)]

def counter(myId, count):
    for i in range(count):
        stdoutmutex.acquire()
        print('[%s] => %s' % (myId, i))
        stdoutmutex.release()
    exitmutexes[myId].acquire() # a child takes a lock

for i in range(10):
    _thread.start_new_thread(counter, (i, 100))

for mutex in exitmutexes:
    while not mutex.locked(): pass # whether all locks have been taken
print('Main thread exiting.')

# alternative with global list of shared integers

stdoutmutex = _thread.allocate_lock()
exitmutexes = [False] * 10

def counter(myId, count):
    for i in range(count):
        stdoutmutex.acquire()
        print('[%s] => %s' % (myId, i))
        stdoutmutex.release()
    exitmutexes[myId] = True

for i in range(10):
    _thread.start_new_thread(counter, (i, 100))


while False in exitmutexes: pass
print('Main thread exiting.')


# the queue module
import queue

stdoutmutex = _thread.allocate_lock()
numthreads = 5
exitmutexes = [_thread.allocate_lock() for i in range(numthreads)]

def counter(myId, count, mutex):
    for i in range(count):
        time.sleep(1/(myId + 1))
        with mutex:
            print('[%s] => %s' % (myId, i))
    exitmutexes[myId].acquire()

for i in range(numthreads):
    _thread.start_new_thread(counter, (i, 5, stdoutmutex))

while not all(mutex.locked() for mutex in exitmutexes): time.sleep(0.25)
print('Main thread exiting.')


# threading module
import threading

class myThread(threading.Thread):
    def __init__(self, myId, count, mutex):
        self.myId = myId
        self.mutex = mutex
        self.count = count
        threading.Thread.__init__(self)

    def run(self): # run() provides thread logic
        for i in range(self.count):
            with self.mutex:
                print('[%s] => %s' % (self.myId, i))

stdoutmutex = threading.Lock()

threads = []
for i in range(10):
    thread = myThread(i, 100, stdoutmutex)
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join() # the join() method waits until the thread exits
print('Main thread exiting')

# example 2 for threading

def action(i):
    print(i ** 32)

class myThread(threading.Thread):
    def __init__(self, i):
        self.i = i
        threading.Thread.__init__(self)

    def run(self): # run() provides thread logic
        print(self.i ** 32)

myThread(2).start()






