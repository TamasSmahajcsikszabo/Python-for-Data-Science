while True:
    reply = input('Enter input:')
    if reply == 'stop': break
    print(reply.upper())

while True:
    reply = input('Enter input:')
    if reply == 'stop': 
        break
    elif not reply.isdigit():
        print(reply.upper())
    else:
        print(int(reply) * 2)
print('Bye!')

while True:
    reply = input('Enter input:')
    if reply == 'stop': break
    try:
        num = int(reply)
    except:
        print(reply.upper())
    else:
        if num < 20:
            print('Low')
        else:
            print(num * 2)
print('Bye!')