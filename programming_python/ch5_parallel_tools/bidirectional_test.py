import sys, os

def spawn(prog, *args):
    stdinFd = sys.stdin.fileno()
    stdoutFd = 1

    parentStin, childStout = os.pipe()
    parentStout, childStin = os.pipe()
    # copy this process
    pid = os.fork()

    if pid:
        os.close(childStin)
        os.close(childStout)
        os.dup2(parentStin, stdinFd)
        os.dup2(parentStout, stdoutFd)
    else:
        os.close(parentStin)
        os.close(parentStout)

        # copies all system information associated with arg1 to arg2
        os.dup2(childStin, stdinFd)
        os.dup2(childStout, stdoutFd)
        args = (prog, ) + args
        os.execvp(prog, args)
        assert False, 'execvp failed!'

if __name__ == "__main__":
    mypid = os.getpid()
    spawn('python3', 'pipes-testchild.py', 'spam')

    print('Hello 1 from parent', mypid)
    sys.stdout.flush()
    reply = input()
    sys.stderr.write('Parent got: "%s"\n' % reply)

    print('Hello 2 from parent', mypid)
    sys.stdout.flush()
    reply = sys.stdin.readline()
    sys.stderr.write('Parent got: "%s"\n' % reply[:-1])

