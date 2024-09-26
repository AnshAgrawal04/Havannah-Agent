import numpy as np

def convert_to_us(sz, state):
    new_state = np.zeros((2*sz-1, 2*sz - 1))
    for i in range(2*sz - 1):
        for j in range(2*sz - 1):
            if abs(i - j) >= sz:
                new_state[(i, j)] = 3

    for i in range(2*sz - 1):
        for j in range(sz):
            new_state[(i, j)] = state[(i, j)]

    for i in range(2*sz - 1):
        for j in range(sz, 2*sz-1):
            if abs(i - j) >= sz:
                continue
            diff = j - sz + 1
            new_state[(i, j)] = state[(i - diff, j)]
    
    return new_state

def convert_to_ta(state):
    sz = 6
    new_state = np.zeros((2*sz - 1, 2*sz - 1))
    for i in range(2*sz - 1):
        for j in range(2*sz - 1):
            if i - j >= sz or i + j > 3*sz - 3:
                new_state[(i, j)] = 3

    for i in range(2*sz - 1):
        for j in range(sz):
            new_state[(i, j)] = state[(i, j)]

    for i in range(2*sz - 1):
        for j in range(sz, 2*sz-1):
            if abs(i - j) >= sz:
                continue
            diff = j - sz + 1
            new_state[(i - diff, j)] = state[(i, j)]
            
    return new_state

state = np.loadtxt('size6.txt')
new_state = convert_to_us(6, state)
print(new_state)
print(convert_to_ta(new_state))