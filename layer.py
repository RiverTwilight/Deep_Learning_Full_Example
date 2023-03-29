class MulLayer:
    def __init__(self) -> None:
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
        
    def backward(self, dout):
        dx = dout * self.x
        dy = dout * self.y

        return dx, dy
