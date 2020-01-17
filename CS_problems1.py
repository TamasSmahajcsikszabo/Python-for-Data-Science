# fibonacci sequence
def fib(n = 40):
    seq = [0, 1]
    ind = list(range(3, n))
    for i in ind:
        new_value = seq[i-3] + seq[i-2]
        seq = seq.append(new_value)
    return seq
i = 3
seq
fib()

# recursive attempt
def fib1(n: int) -> int:
    return fib1(n - 1) + fib1(n - 2)

if __name__ == "main":
    print(fib1(5))

def fib2(n: int) -> int:
    if n < 2: # base case
        return n
    return fib2(n - 2) + fib2(n - 1) #recursive case
print(fib2(5))
if __name__ == "__main__":
    print(fib2(5))
    print(fib2(10))

print(fib2(40))

# memoization of the results (Donald Michie)
# memoization with a python library
from typing import Dict
memo: Dict[int, int] = {0: 0, 1: 1}

def fib3(n: int) -> int:
    if n not in memo:
        memo[n] = fib3(n-1) + fib3(n-2)
    return memo[n]
print(fib3(3))
print(fib3(10))
print(fib3(20))
print(fib3(50))

# memoization with the python built-in decorator
from functools import lru_cache

@lru_cache(maxsize=None)
def fib4(n: int) -> int:
    if n < 2:
        return n
    return fib4(n-2) + fib4(n-1)

print(fib4(50))

# with iteration
def fib5(n: int) -> int:
    if n ==0: return n
    last: int = 0
    next: int = 1
    for _ in range(1, n):
        last, next = next, last + next
    return next 

print(fib5(50))

# generator option
from typing import Generator
def fib6(n: int) -> Generator[int, None, None]:
    yield 0
    if n > 0: yield 1
    last: int = 0
    next: int = 1
    for _ in range(1, n):
        last, next = next, last + next
        yield next 

for i in fib6(50):
    print(i)

# compression
def _compress(self, gene: str) -> None:
    self.bit_string: int = 1
    for nucleotide in gene.upper():
        self.bit_string <<= 2
        if nucleotide == "A":
            self.bit_string |= 0b00
        elif nucleotide == "C":
            self.bit_string |= 0b01
        elif nucleotide == "G":
            self.bit_string |= 0b10
        elif nucleotide == "T":
            self.bit_s
tring |= 0b11
        else:
            raise ValueError("Invalid Nucleotide:{}".format(nucleotide))


class compressed_gene:
    def __init__(self, gene: str) -> None:
        self._compress(gene)

def decompress(self) -> str:
    gene: str = ""
    for i in range(0, self.bit_string.bit_length() - 1, 2):
        bits: int = self.bit_string >> i & 0b11
        if bits == 0b00:
            gene += "A"
        elif bits == 0b01:
            gene += "C"
        elif bits == 0b10:
            gene += "A"
        elif bits == 0b11:
            gene += "T"
        else:
            raise ValueError("Invalid bits:{}".format(bits))
    return gene[::-1]

def __str__(self) -> str:
    return self.decompress()

from sys import getsizeof
original: str =  "TAGGGATTAACCGTTATATATATATAGCCATGGATCGATTATATAGGGATTAACCGTTATATATATATATCCATGGATCGATTATA" * 100
print("original is {} bytes".format(getsizeof(original)))
compressed = compressed_gene(gene = original)
print("Compressed is {} bytes".format(getsizeof(compressed.bit_string())))
print(compressed)
print("original and decompressed are the same: {}".format(original == compressed.decompress()))

compressed_gene("T")
