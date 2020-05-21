# 1. Fibonacci
from typing import TypeVar, Generic, List


def fib_n(n: int) -> int:
    start: int = 0
    end: int = 1
    for i in range(3, n + 1):
        start, end = end, start + end
    return end


fib_n(7)
fib_n(11)

# 2. ergonomic wrapper


class into_bitstring:
    def __init__(self, input: int) -> int:
        self.data = input

    def __getitem__(self) -> None:
        self.data <<= 2


g = into_bitstring(23223234)
for i in g:
    print(i)
# 3. Hanoi

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._container: List[T] = []

    def push(self, item: T) -> None:
        self._container.append(item)

    def pop(self) -> T:
        return self._container.pop()

    def __repr__(self) -> str:
        return repr(self._container)


num_discs: int = 4
tower_a: Stack[int] = Stack()
tower_b: Stack[int] = Stack()
tower_c: Stack[int] = Stack()
tower_d: Stack[int] = Stack()

def hanoi(begin: Stack[int], end: Stack[int], temp: list, n: int) -> None:
if n == 1:
        end.push(begin.pop())
else:
    hanoi(begin, temp, end, n - 1)
    for t in temp:
        hanoi(begin, end, temp[t], 1)
    hanoi(temp, end, begin, n - 1)


