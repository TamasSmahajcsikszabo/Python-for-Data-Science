import signal
import sys
import time


def now(): return time.ctime(time.time())


def onSignal(signum, stackframe):
    print('Got signal', signum, 'at', now())


signum = int(sys.argv[1])
# install arg2 function to handle arg1 signal when raised
signal.signal(signum, onSignal)
while True:
    signal.pause()
