# A stack is a data structure that is modeled on the concept of Last-In-First-Out (LIFO). The last thing put into it is the first thing that comes out of it.

from typing import TypeVar, Generic, List
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


hanoi(tower_a, tower_d, list(tower_b, tower_c), num_discs)
print(tower_a)
print(tower_b)
print(tower_c)
print(tower_d)
