import _thread
import time
import queue
import threading

numconsumers = 4
numproducers = 4
nummessages = 4

safeprint = _thread.allocate_lock()
# the queue is assigned to a global variable, i.e.shared by all threads
dataQueue = queue.Queue()


def producer(idnum):
    for msg in range(nummessages):
        time.sleep(idnum)
        dataQueue.put('[producer id=%d, count=%d]' % (idnum, msg))


def consumer(idnum):
    while True:
        time.sleep(0.1)
        try:
            data = dataQueue.get(block=False)
        except queue.Empty:
            pass
        else:
            with safeprint:
                print('consumer', idnum, 'got=>', data)


if __name__ == "__main__":
    starttime = time.time()
    for i in range(numconsumers):
        # _thread.start_new_thread(consumer, (i,))
        thread = threading.Thread(target=consumer, args=(i,))
        thread.daemon = True  # to make sure they don't prolong runtime
        thread.start()

    waitfor = []
    for i in range(numproducers):
        thread = threading.Thread(target=producer, args=(i,))
        waitfor.append(thread)
        thread.start()

    # time.sleep((numproducers - 1) * nummessages + 1)
    for thread in waitfor:
        thread.join()
    endtime = time.time()
    runtime = endtime - starttime
    print('Main thread exiting under {0} sec.'.format(round(runtime, 3)))
