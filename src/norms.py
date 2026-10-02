import numpy as np

def L2norm(x):
    s = 0
    for i in x:
        s += i*i
    return np.sqrt(s)
