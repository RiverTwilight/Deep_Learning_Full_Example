import numpy as np
import matplotlib.pyplot as plt

# Create an array & visit
a = np.array([1, 2, 3, 4, 5])
print(a)
print(a[3])
print(a[np.array([2, 3])])

# Create a 2D array & boardcast
b = np.array([[1, 2], [3, 4]]) 
print(b * 2)
print(b * np.array([3, 3]))

# Create 2D function graph
x = np.arange(0, 6, 0.05)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, label="cos", linestyle = "--")
plt.title("sin & cos")
plt.show()
