import numpy as np

class Relu:
    def __init__(self) -> None:
        self.mask = None
    
    def forward(self, x):
        """
        x should be a numpy array here
        """
        self.mask = (x <= 0) # An array represting wheather each element is larger than 0. [True, False, False]
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        """
        Set all the `Ture` in mask to 0
        """
        dout[self.mask] = 0
        dx = dout

        return dx

reluTest = Relu()

print(reluTest.forward(np.array([2, 1, -1, -10, 0]))) # 2 1 0 0 0
print(reluTest.mask) # [False False  True  True  True]
print(reluTest.backward(np.array([2, 1, -1, -10, 0]))) # 2 1 0 0 0

class Sigmoid:
    def __init__(self) -> None:
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x)) # 1 / (1 + e^(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine:
    def __init__(self) -> None:
        pass

class Softmax:
    """
    Make sure the sum of output is 1
    """
    def __init__(self) -> None:
        pass
