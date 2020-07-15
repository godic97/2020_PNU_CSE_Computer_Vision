import numpy as np

def fh(x, W, b):
    return W * x + b


x = np.array([2, 4, 5, 7])
y = np.array([9, 5, 6, 2])

W = 1
b = 1
a = 0.01

step = 10000


for i in range(0, step):
    for j in range(0, 4):
        h = fh(x[j], W, b)
        b_tmp = b + a * (y[j] - h)
        W_tmp = W + a * (y[j] - h) * x[j]

        W = W_tmp
        b = b_tmp

print("y = ", W,"x + ", b)

