a, *b = "spam" # b is assigned to the rest of the string sequence
spam, ham = 'yum', 'YUM'

# sequence assignments - the length have to match, but the types are on me
(a,b,c) = "ham"
[a,b,c] = (1,2,3)
string = "SPAM"
(a,b,c) = list(string[:2]) + [string[2:]]
seq = [1,2,3,4]
a, *b, c = seq

L = [1,2,3,4]
while L:
    front, *L = L
    print(front, L)

# extended sequence unpacking always returns a list, even empty
