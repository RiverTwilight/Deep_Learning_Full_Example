# Activation Function

import numpy as np

def step(x):
    y = x > 0
    return y.astype(int)

print(step(np.array([1, 3, 0])))

# Old and widely-used activation function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)
