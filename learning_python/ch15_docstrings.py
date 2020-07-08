"""
Chessboard generator funcion
Simply define the length of the table size
"""
def chessboard():
    size = int(input("Please indicate board size!"))
    black = '#'
    white = ' '
    rows = range(1,size+1)
    board = ''
    for r in rows:
        line = ''
        index = r % 2 + 1
        while len(line) < size:
            if index % 2 != 0:
                line = line + white
                index += 1
            else:
                line = line + black
                index += 1
        board = board + line + '\n'
    print(board)

chessboard()

L = [1,2,3,4,5]
L.append(6)
L

s = 'example'

for i in s: 
    print(ord(i), end = ' ')


z = int()
for i in s:
   z = z + ord(i)
print(z)

L = []
for i in s:
    L.append(ord(i))

list(map(ord, s))

[ord(c) for c in s]

for i in range(50):
    print('hello %d\n\a' % i)

a = ['a', 'c', 'b', 'd']
b = [1,2,3,4]

D = {x : y for x,y in zip(a,b)}

K = list(D.keys())
K.sort()
for k in K:
    print(k, '=>', D[k])

L = [1,2,4,8,16,32,64]
X = 5
i = 0
while i < len(L): 
    if 2 ** X == L[i]:
        print('at index', i)
        break
    i +=1
else:
    print(X, 'not found')

for c in L:
    if 2 ** X == c:
        print('found at index', L.index(c))
    else:
        print('not found')

if (2 ** X) in L:
    print((2 ** X), 'found at index', L.index(2 ** X))
else:
    print('not found')

L = []
for i in range(0,8,1):
    L.append(2**i)

if (2 ** X) in L:
    print((2 ** X), 'found at index', L.index(2 ** X))
else:
    print('not found')

[2 ** i for i in range(0,8,1)]