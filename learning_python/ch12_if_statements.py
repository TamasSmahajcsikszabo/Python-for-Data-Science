# compound statements = statements containing other statements, control-flow statements
# there is no switch or case functionality

# dynamic construction of a multiway branch instead of hard-coded if:
choice = 'ham'
print({
    'ham': 11,
    'spam' :22,
    'spider' : 22}[choice])

# the else term can be modelled by using the get method or the try statement
branch = {
    'ham': 11,
    'spam' :22,
    'spider' : 22}

print(branch.get('ham', 'bad choice'))
print(branch.get('antilope', 'bad choice'))

# the same with if and in
choice = 'elephant'
if choice in branch:
    print(branch[choice])
else: print('Bad choice')

# adding functions into dictionaries
def function(): ...
def default(): ...
branch = {
    'spam': lambda: ...,
    'ham': function, 
    'eggs': lambda: ...}

branch.get(choice, default)()

number = range(1,100)

for n in number: 
    if n % 5 == 0 and n % 3 != 0:
        print('buzz')
    elif n % 5 != 0 and n % 3 == 0:
        print('fizz')
    elif n % 5 == 0 and n % 3 == 0:
        print('fizzbuzz')
    else: print(n)

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