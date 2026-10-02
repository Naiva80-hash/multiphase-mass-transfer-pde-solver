import numpy as np
def num_jacobe_colored(f, x, Fx, rows_by_col, groups, rel_step=1e-7):
    x = np.asarray(x, float)
    Fx = np.asarray(Fx, float)
    n = x.size
    m = Fx.size

    J = np.zeros((m, n), float)
    h = rel_step * (1.0 + np.abs(x))

    x1 = x.copy()
    for cols in groups:
        x1[cols] = x[cols] + h[cols]
        F1 = np.asarray(f(x1), float)
        dF = F1 - Fx

        # fill each column only on its affected rows
        for c in cols:
            r = rows_by_col[c]
            J[r, c] = dF[r] / h[c]


        x1[cols] = x[cols]  # restore

    return J