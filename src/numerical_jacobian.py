import numpy as np

def num_jacobe_fast(f, x, Fx=None, rel_step=1e-7):
    x = np.asarray(x, float)
    if Fx is None:
        Fx = np.asarray(f(x), float)

    n = x.size
    m = Fx.size
    J = np.zeros((m, n), float)

    h = rel_step * (1.0 + np.abs(x))

    for i in range(n):
        x1 = x.copy()
        x1[i] += h[i]
        F1 = np.asarray(f(x1), float)
        J[:, i] = (F1 - Fx) / h[i]

        v = np.max(abs(F1 - Fx))
        if v == 0:
            print(f"Input {i} from {n} inputs will not disturb the function")
    
    return J
