import numpy as np


def grover(states):
    """Performs one Grover iteration on the input states"""
    search_value = 2
    states[search_value] *= -1
    print(states)
    x0 = np.mean(states)
    print(x0)
    states = 2*x0 - states
    return states


states = 1/(np.sqrt(32))*np.ones(32)
search_value = 2

print(grover(grover(grover(grover(states))))[2]**2)
