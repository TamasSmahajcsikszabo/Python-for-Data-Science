def fib_n(n: int) -> int:
    start: int = 0
    end: int = 1
    for i in range(3, n + 1):
        start, end = end, start + end
    return end

fib_n(7)
fib_n(11)

def encoded(n: int) -> bytes:
    return n.encode()
