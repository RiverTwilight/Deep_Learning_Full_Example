# Perceptron algorithm
# It is a simplified model of biological neurons and is the building block of artificial neural networks.

import numpy as np


# Only when both x1 and x2 is 1, the output is 1.
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2 # When meet the requirement, it should be 1 now.
    if tmp <= theta: # Note this equal to 'tmp - theta <= 0'
        return 0
    elif tmp > theta:
        return 1

# Use numpy
def AND_2(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.7
    tmp = np.sum(x * w) - b
    if tmp <= 0:
        return 0
    else:
        return 1

print(AND_2(1, 1))
print(AND_2(1, 0))

# Only when both x1 and x2 is 1, the output is 0.
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b # Here diffrenet from AND
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.3
    tmp = x1*w1 + x2*w2 
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print(OR(1, 0))
print(OR(0, 0))

# A two-layer perceptron
def XOR(x1, x2):
    y1 = OR(x1, x2)
    y2 = NAND(x1, x2)
    return AND_2(y1, y2)

print(XOR(1, 0)) # 1
print(XOR(0, 0)) # 0

# You've learned all basic of neural network. Now you can continue read neural-network.py .
