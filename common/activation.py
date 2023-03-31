import numpy as np
from neural_network import cross_entropy_error, softmax

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
    def __init__(self, W, b) -> None:
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b # Boardcasting...

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftmaxWithLoss:
    """
    Make sure the sum of output is 1
    """
    def __init__(self) -> None:
        self.loss = None
        self.y = None
        self.x = None

    def forward(self, x, t):
        self.t = t 
        # Teaching Data. Marking the right answer.
        # Set right anwser to 1 and wrongs to 0. For exmaple, [0, 0, 0, 1, 0, 0]

        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

print(SoftmaxWithLoss().forward(np.array([0.0001, 0.9999, 0.00001]), np.array([0, 1, 0]))) # 0.5515 When we got accurency data, It's very small
print(SoftmaxWithLoss().forward(np.array([0.9, 0.05, 0.05]), np.array([0, 1, 0]))) # 1.4677 When we got coraouse data, It's very large
