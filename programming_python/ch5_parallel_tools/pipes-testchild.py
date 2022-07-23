import os
import time
import sys

mypid = os.getpid()
parent = os.getppid()
sys.stderr.write(
    'Child %d of %d got arg: "%s"\n' %
    (mypid, parent, sys.argv[1]))

for i in range(2):
    time.sleep(3)
    recv = input()
    time.sleep(4)
    send = 'child %d got: [%s]' % (mypid, reply)
    print(send)
    sys.stdout.flush()
