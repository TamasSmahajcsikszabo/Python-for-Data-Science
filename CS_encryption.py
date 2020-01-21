# one-time pad encryption
string = "stardestroyer"
s = string.encode()
type(s)

# a pseudo-random data generating function
from secrets import token_bytes
from typing import Tuple

def random_key(length: int) -> int:
    tb: bytes = token_bytes(length)
    return int.from_bytes(tb, "big")
