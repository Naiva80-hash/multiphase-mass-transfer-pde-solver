from math import exp
import numpy as np


#Func inside the integral for CWg component
def g(cwk, cethk, i, j, k, indicies_dict_sph_w, indicies_dict_sph_eth):
    l_w = indicies_dict_sph_w.get((i, j, k))
    l_e = indicies_dict_sph_eth.get((i, j, k))
    cw = cwk[l_w] if l_w is not None else 0.0
    ce = cethk[l_e] if l_e is not None else 0.0
    den = cw + ce + 1e-10

    if (den <= 0.0) or (cw < 0.0) or (ce < 0.0) or (cw/den > 1.2) or (cw/den < -0.2):
        #print("BAD xw at (i,j,k)=", (i,j,k), " l=", l,
        #      " cw=", cw, " ceth=", ce, " den=", den, " xw=", cw/den)
        xw = cw / den
        #raise RuntimeError("Nonphysical composition/denominator issue")

    xw = cw / den
    return xw * exp((0.9146 + 1.0736*xw)*(1-xw)**2)

#Func inside the integral for CWg component
def g2(cwk, cethk, i, j, k, indicies_dict_sph_w, indicies_dict_sph_eth):
    l_w = indicies_dict_sph_w.get((i, j, k))
    l_e = indicies_dict_sph_eth.get((i, j, k))
    cw = cwk[l_w] if l_w is not None else 0.0
    ce = cethk[l_e] if l_e is not None else 0.0
    den = cw + ce + 1e-10
    xw = cw / den
    return (1-xw)*exp((1.4514 - 1.0736*(1-xw))*xw**2)


