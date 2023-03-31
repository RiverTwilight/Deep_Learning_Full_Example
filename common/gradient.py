import numpy as np

def numerical_gradient(f, x):
    """
    Gradient points a direction which the output of a function desend fastest.
    """
    h = 1e-4
    grad = np.zeros_like(x) # Generate an array which has same shape with x

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val

    return grad

def test_function(x):
    return x[0] ** 2 + x[1] ** 2

def gradient_desent(f, init_x, lr=0.01, step_num=100):
    """
    lr is Learning Rate. This should not be too large or too small.
    """

    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

print(numerical_gradient(test_function, np.array([3.0, 4.0])))
