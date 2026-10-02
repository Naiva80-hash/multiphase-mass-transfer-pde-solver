import numpy as np

def init(N_tot_w, N_tot_eth, P_bar=1*1.01325, T=35+273.15, R=8.314462618,
         xw0=0.25, xeth0=0.75, C_L_tot=4e4):

    cw0   = np.ones((N_tot_w,),   dtype=float) * (xw0   * C_L_tot)
    ceth0 = np.ones((N_tot_eth,), dtype=float) * (xeth0 * C_L_tot)

    P_Pa = P_bar * 1e5
    Ctot0 = P_Pa / (R * T)

    # sealed container starts as nitrogen only
    cn2g0  = np.array([Ctot0], dtype=float)
    cwg0   = np.array([1e-12],   dtype=float)
    cethg0 = np.array([1e-12],   dtype=float)

    return np.concatenate((cw0, ceth0, cn2g0, cwg0, cethg0))



def init_xk(x, N_tot_w, N_tot_eth):
    """
    This func does 3 things:
    1-Make a safe copy in float
    2-Locate gas block
    3-clamp only the gas vars to be nonnegative
    """
    x = np.asarray(x, float).copy()
    gas0 = N_tot_w + N_tot_eth
    x[gas0:gas0+3] = np.maximum(x[gas0:gas0+3], 0.0)
    return x
