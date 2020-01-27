# one-time pad encryption
# endoing and decoding a string into bytes and back
string = "stardestroyer"
s = string.encode()
type(s)
s.decode() == string

# The dummy data must be the same length as the original data, truly random, and completely secret.
# a pseudo-random data generating function
from secrets import token_bytes
from typing import Tuple

def random_key(length: int) -> int:
    tb: bytes = token_bytes(length)
    return int.from_bytes(tb, "big")

random_key(12)

# the exclusive or bitwise operation
1 ^ 0
1 ^ 1

a = 123
b = 234
c = a ^ b
c ^ b == a

bytes(a)
bytes(b)
bytes(c)

def encrypt(original: str) -> Tuple[int, int]:
    original_bytes: bytes = original.encode()
    dummy: int = random_key(len(original_bytes))
    original_key: int = int.from_bytes(original_bytes, "big")
    encrypted: int = original_key ^ dummy
    return dummy, encrypted

def decrypt(key1: int, key2: int) -> str:
    decrypted: int = key1 ^ key2
    temp: bytes = decrypted.to_bytes(decrypted.bit_length() + 7 // 8, "big")
    return temp.decode()

