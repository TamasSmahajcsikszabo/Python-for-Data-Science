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
