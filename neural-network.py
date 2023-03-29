# Activation Function

import numpy as np
import matplotlib.pyplot as plt

def step(x):
    y = x > 0
    return y.astype(int)

print(step(np.array([1, 3, 0])))

# Old and widely-used activation function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# There are three common places of these three activation function:
#
#  1. The output is between 0 to 1
#  2. Both is liner function
#  3. The more important the input is, the bigger the output is.

x = np.array([1, 2])
w = np.array([[3, 4], [5, 2]]) # The row number should equal to x's length.

# Diffrent operation order will output diffrenet result
print(np.dot(w, x)) # [11, 9]
print(np.dot(x, w)) # [13, 8] [1 x 3 + 2 x 5, 1 x 4 + 2 x 4]

# Central Difference Derivation
# We use 2 h to reduce the deviation.
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def func_1(x):
    return 0.01 * x ** 2 + 0.1 * x

x = np.arange(0.0, 20.0, 0.1)
y = numerical_diff(func_1, x) # This is a valid operation (boardcast)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)
plt.show()
