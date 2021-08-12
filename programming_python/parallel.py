# fork processes
# ~ forking is like copying a program in memory and running it parallel
# ~ the original is the parent process, while the others are the children
# ~ all run independently children can change from and outlive parents

from multiprocessing.pool import ThreadPool
import threading
import queue
import _thread
import os
import time
from mazes import Maze

maze = Maze(
    10,
    10,
    indicator="maze",
    name="maze",
    cellsize=3,
    algorithm="sidewinder")


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
# shared global memory space with process
# global variable interchargebility


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

# at forks, children live after parents, here not! So the main thread has
# to wait for the rest to finish
time.sleep(6)
print('Main thread exiting.')

# namespaces and objects are shared among all threads
# one object change can be seen by all, main or child alike
# two threads changing one object at the same time might lead to object
# corruption
# _thread achieves this with *locks*


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
    exitmutexes[myId].acquire()  # a child takes a lock


for i in range(10):
    _thread.start_new_thread(counter, (i, 100))

for mutex in range(len(exitmutexes)):
    print('Mutex {0} locked status is {1}'.format(mutex, exitmutexes[mutex].locked()))
    while not exitmutexes[mutex].locked():
        pass  # whether all locks have been taken
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


while False in exitmutexes:
    pass
print('Main thread exiting.')

# alternative with 'with' statement
# 'with': the lock's context manager will acquire the lock on the code block,
# and will release it once run no matter what exceptions are met

stdoutmutex = _thread.allocate_lock()
numthreads = 5
exitmutexes = [_thread.allocate_lock() for i in range(numthreads)]


def counter(myId, count, mutex): # shared object of stdoutmutex is passed in
    for i in range(count):
        time.sleep(1/(myId + 1))
        with mutex:
            print('[%s] => %s' % (myId, i)) # no need to call release, 'with' does it
    exitmutexes[myId].acquire()

for i in range(numthreads):
    _thread.start_new_thread(counter, (i, 5, stdoutmutex))

while not all(mutex.locked() for mutex in exitmutexes): time.sleep(0.25)
print('Main thread exiting.')


# threading module

class myThread(threading.Thread):
    def __init__(self, myId, count, mutex):
        self.myId = myId
        self.mutex = mutex
        self.count = count
        threading.Thread.__init__(self)

    def run(self):  # run() provides thread logic and the thread's action
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
    thread.join()  # the join() method waits until the thread exits
print('Main thread exiting')



# example 2 for threading
# 1. subclass case, prefereed for OOP and if per-thread information is needed
def action(i):
    print(i ** 32)

class myThread(threading.Thread):
    def __init__(self, i):
        self.i = i
        threading.Thread.__init__(self)

    def run(self):  # run() provides thread logic
        print(self.i ** 32)
myThread(2).start()

# 2. pass in action to the 'target' (default Thread class) argument, i.e.
# without subclasses
thread = threading.Thread(target=(lambda: action(2)))
thread.start()

# synchronizing access to shared objects
# without synchronizing:
def not_synched():

    def adder():
        global count
        count = count + 1
        time.sleep(0.5)
        count = count + 1

    threads = []
    for i in range(100):
        thread = threading.Thread(target = adder, args=())
        thread.start()
        threads.append(thread)

    for thread in threads: thread.join()
    print(count)

count = 0
not_synched()
# even statements such as assignments are not guaranteed to run to completion
# by themselves, i.e. they are not atomic

# apply thread locks to synchronize:

def foo(word, number):
    print(word * number)
    return number


words = ['hello', 'world', 'test', 'word', 'another test']
numbers = [1, 2, 3, 4, 5]
pool = ThreadPool(5)
results = []
for i in range(0, len(words)):
    results.append(pool.apply_async(foo, args=(words[i], numbers[i])))

pool.close()
pool.join()
results = [r.get() for r in results]
print results


# the queue module

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

while not all(mutex.locked() for mutex in exitmutexes):
    time.sleep(0.25)
print('Main thread exiting.')
