import numpy as np
from math import pi, log, exp, cos, sin
from initial_conditions import *
from interface_thermodynamics import *
from broyden_solver import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DBG_FLUX_EVERY = 0
dbg_flux_calls = 0
dbg_flux2_calls = 0
DBG_ANG_EVERY = 0
#Constants
#Water
aw = -7.76451
bw = 1.45838
cw = -2.77580
dw = -1.23303
Tcw = 647.3
Pcw = 221.2
MWw = 18.015
neww = 12.22
#Ethanol
aeth = -8.51838
beth = 0.34163
ceth = -5.73683
deth = 8.32581
Tceth = 513.9
Pceth = 61.4
MWeth = 46.069
neweth = 51.77
#N2
MWn2 = 28.013
newn2 = 9.08
#Whole system
T = 35 + 273.15 #K
eps = 0.3 
P = 1 * 1.01325 #Total pressure in the cylinder
#Wagner Eq
def P_sat_cal(A, B, C, D, Pc, T, Tc):
    Tr = T / Tc
    tau = 1.0 - Tr
    return Pc * np.exp((A*tau + B*tau**1.5 + C*tau**3 + D*tau**6) / Tr)


def MWij_cal(MWi, MWj):
    MWij = 2/(1/MWi+1/MWj)
    return MWij

#Binary Liquid mixture
def Dij_cal(T, P, neui, neuj, MWij):
    Dij = 0.00143*T**1.75/(P * (neui**(1/3)+neuj**(1/3))**2*np.sqrt(MWij))
    return Dij

def activity_coeffs(xw, xeth):
    gamma_w = exp((0.9146 + 1.0736*xw)*(1-xw)**2)
    gamma_eth = exp((1.4514-1.0736*xeth)*xw**2)
    return gamma_w, gamma_eth

#Assuming gamma_i will not change based on liquid compositions

M_weth = MWij_cal(MWw, MWeth)
M_ethw = MWij_cal(MWeth, MWw)
M_n2eth = MWij_cal(MWn2, MWeth)
M_n2w = MWij_cal(MWn2, MWw)
M_wn2 = M_n2w
M_weth = M_ethw
M_ethn2 = M_n2eth
M_ethw = M_weth
#m2/s unit
DW_eth = Dij_cal(T, P, neww, neweth, M_weth)*1e-4
Deth_W = Dij_cal(T, P, neweth, neww, M_ethw)*1e-4
Dn2_eth = Dij_cal(T, P, newn2, neweth, M_n2eth)*1e-4
Dn2_W = Dij_cal(T, P, newn2, neww, M_n2w)*1e-4
DW_n2 = Dij_cal(T, P, neww, newn2, M_wn2)*1e-4
DW_eth = Dij_cal(T, P, neww, neweth, M_weth)*1e-4
Deth_n2 = Dij_cal(T, P, neweth, newn2, M_ethn2)*1e-4
Deth_w = Dij_cal(T, P, neweth, neww, M_ethw)*1e-4
Psat_w   = P_sat_cal(aw,  bw,  cw,  dw,  Pcw,  T, Tcw)
Psat_eth = P_sat_cal(aeth,beth,ceth,deth,Pceth,T, Tceth)
D_liq = 1.25e-9 #m2/s
D_eff_w = eps * D_liq
D_eff_eth = eps * D_liq


#Sphere meshing
Rc = 0.15
r_sph = 0.25 * Rc
tetha_max = 0.98*pi
phi_max = 2*pi
N_R_BASE = 19
N_R_w = N_R_BASE
N_R_eth = N_R_BASE
N_Tetha = 5
N_Phi = 5



# Cluster nodes near r=r_sph while keeping the rest of the style unchanged.
tighten_r_surface = 3.0  # >1 clusters nodes near r=r_sph, =1 uniform


# Radial grid from 0 to r_max with clustering near the surface
# If p <= 1 returns uniform linspace
# If p > 1 maps parameter s through r which places more points near r=r_max
# When s → 1, r → r_max..If we took derviative from r w.r.t s, we have 
# dr/ds = r_max*p*(1-s)^(p-1), when s → 1, dr/ds → 0. As a consequence, there are lots of
# points near r=r_max, and less points around r=0 as dr/ds = r_max*p
# for p>1, (1-s)^(p-1) collapse fast near r=r_max
def stretched_r_points(r_max, n, p):
    if p <= 1.0:
        return np.linspace(0.0, r_max, n+1, endpoint=True)
    s = np.linspace(0.0, 1.0, n+1, endpoint=True)
    return r_max * (1.0 - (1.0 - s)**p)


r_points = stretched_r_points(r_sph, N_R_BASE, tighten_r_surface)
tetha_points = np.linspace(0, pi, N_Tetha+1, endpoint=True)

diff_tetha = np.diff(tetha_points)[0]
phi_points = np.linspace(0, phi_max, N_Phi+1, endpoint=True)
diff_phi = np.diff(phi_points)[0]
#Cylinder meshing
R_cyldr= 0.15
H = 0.4


def Vg_cylinder_minus_sphere(Rc, H, r_sph):
    """
      Gas volume in the tank
    """
    return (np.pi * Rc**2 * H) - (4.0/3.0) * np.pi * r_sph**3

Vg_const = Vg_cylinder_minus_sphere(Rc, H, r_sph)


def total_moles_liquid_species_FD(x_vec, species,
                                 N_R_w, N_R_eth, N_Tetha, N_Phi,
                                 r_points, tetha_points, phi_points,
                                 ind_w, ind_e, N_tot_w, N_tot_eth,
                                 theta_max=None):
    """
    Compute this intergal 
    n_i_liq = int_phi(int_tetha(int_r (ci(r, tetha, phi) r^2 sin(tetha)d_r.d_tetha.d_phi))))
    using trapezodial-like quadrature on the grid nodes
    """

    # Pick which field to integrate
    # Unknow vector is x=[cw(all nodes), ceth(all nodes), c_gas]
    if species.lower() == "water":
        N_R = N_R_w
        ind = ind_w
        offset = 0
    elif species.lower() == "ethanol":
        N_R = N_R_eth
        ind = ind_e
        offset = N_tot_w
    else:
        raise ValueError("species must be 'water' or 'ethanol'")

    if N_R < 0:
        return 0.0

    # Computing grid
    r = r_points[:N_R+1]
    dr = np.diff(r)
    dth  = tetha_points[1] - tetha_points[0]
    dphi = phi_points[1] - phi_points[0]

    # dr weights attached to node ri
    wi = np.ones(N_R+1)
    if N_R >= 1:
        wi[0] = 0.5 * dr[0]
        wi[-1] = 0.5 * dr[-1]
        if N_R > 1:
            wi[1:-1] = 0.5 * (dr[:-1] + dr[1:])

    n_liq = 0.0

    for i in range(N_R + 1):
        r = r_points[i]

        for j in range(N_Tetha + 1):
            th = tetha_points[j]
            # Integrating from 0 to tetha_max not the full sphere
            if (theta_max is not None) and (th > theta_max):
                continue
            # Ends point in tehta direction get half weight
            # Interior nodes get full weight
            wj = 0.5 if (j == 0 or j == N_Tetha) else 1.0
            sinth = np.sin(th)
            # Intergrating from 0 to 2pi, without including 2pi since it is duplicated
            for k in range(N_Phi):  # exclude duplicated node at N_Phi
                l = ind[(i, j, k)]
                c = x_vec[offset + l]
                n_liq += (wi[i] * wj) * c * (r**2) * sinth * dth * dphi

    return float(n_liq)



def total_moles_gas_species(x_vec, species, Vg, N_tot_w, N_tot_eth):
    """n_i,gas = c_i,gas * Vg"""
    gas_off = N_tot_w + N_tot_eth
    if species.lower() == "water":
        c = float(x_vec[gas_off + 1])  # cwg
    elif species.lower() == "ethanol":
        c = float(x_vec[gas_off + 2])  # cethg
    else:
        raise ValueError("species must be 'water' or 'ethanol'")
    return c * Vg



# Adaptive time step parameters

t_end = 3e4
t_now = 0.0

dt = 1
dt_min = 1e-2
dt_max = 5e5


rtol = 1e-2      
atol = 1e-8     
safety = 0.9
order = 1

it_target = 10
it_hard   = 20
shrink_reject = 0.7   


def build_sphere_mapping(N_R, N_Tetha, N_Phi):
    """
    Mapping function which produces:
    m_sph : Total number of parameters
    indcies_dict_sph : A dictionary that maps (i,j,k) → l (Matrix indicies to vector)
    indcies_dict_sph_rev : A dictionary that maps l → (i, j, k) (Vector indicies to matrix indicies)
    """
    m_sph = (N_R + 1) * (N_Tetha + 1) * (N_Phi + 1)
    indcies_dict_sph = {}
    indcies_dict_sph_rev = {}
    for l in range(0, m_sph):
        k = int(l / ((N_R + 1) * (N_Tetha + 1)))
        p = l % ((N_R + 1) * (N_Tetha + 1))
        i = int(p / (N_Tetha + 1))
        j = p % (N_Tetha + 1)
        indcies_dict_sph[(i, j, k)] = l
        indcies_dict_sph_rev[l] = (i, j, k)
    return m_sph, indcies_dict_sph, indcies_dict_sph_rev

## New mapping for cylinderical coordinates

def rebuild_grid_dependent_state(N_R_w, N_R_eth):
    """
    Rebuilding grids if the concentrations at r = N_R becomes negative.
    We have to R_inf for each of the two species.
    One for water and one for ethanol
    """
    if N_R_w >= 0:
        m_sph_w, indcies_dict_sph_w, indcies_dict_sph_rev_w = build_sphere_mapping(N_R_w, N_Tetha, N_Phi)
        N_tot_w = (N_R_w + 1) * (N_Tetha + 1) * (N_Phi + 1)
    else:
        m_sph_w, indcies_dict_sph_w, indcies_dict_sph_rev_w = 0, {}, {}
        N_tot_w = 0

    if N_R_eth >= 0:
        m_sph_eth, indcies_dict_sph_eth, indcies_dict_sph_rev_eth = build_sphere_mapping(N_R_eth, N_Tetha, N_Phi)
        N_tot_eth = (N_R_eth + 1) * (N_Tetha + 1) * (N_Phi + 1)
    else:
        m_sph_eth, indcies_dict_sph_eth, indcies_dict_sph_rev_eth = 0, {}, {}
        N_tot_eth = 0
    return (
        m_sph_w, N_tot_w, indcies_dict_sph_w, indcies_dict_sph_rev_w,
        m_sph_eth, N_tot_eth, indcies_dict_sph_eth, indcies_dict_sph_rev_eth
    )


m_sph_w, N_tot_w, indcies_dict_sph_w, indcies_dict_sph_rev_w, \
m_sph_eth, N_tot_eth, indcies_dict_sph_eth, indcies_dict_sph_rev_eth = rebuild_grid_dependent_state(N_R_w, N_R_eth)

# Flags for whether each liquid component is still active .
water_active = True
ethanol_active = True

# Initializing vector of unknowns

#Building liquid concentration vectors
cw = np.zeros((m_sph_w, ), dtype = float)
ceth = np.zeros((m_sph_eth,), dtype = float)
#Building gas conentration vectors

cwg = np.zeros((1,), dtype = float)
cethg = np.zeros((1,), dtype = float)
cn2g = np.zeros((1, ), dtype = float)


    
R_gas = 8.314462618  # J/mol/K
P0_bar = P          # 1 bar
Cn2_fixed = (P0_bar * 1e5) / (R_gas * T)   # mol/m^3  (since Vg is constant)


def gas_mole_fracs(Cw, Ceth, Cn2, eps=1e-30):
    """
    produce gas mole fractions
    """
    Ct = Cw + Ceth + Cn2
    Ct = Ct if Ct > eps else eps
    return Cw/Ct, Ceth/Ct, Cn2/Ct, Ct

def D_mix_3(yj, yk, Dij, Dik, eps=1e-30):
    """
    mixture-averaged D_im for species i in (i,j,k)
    """
    return 1.0 / (yj/max(Dij, eps) + yk/max(Dik, eps) + eps)




def int_phi_tetha_FV(gfun, cwk_vec, cethk_vec, r_sph,
                     tetha_points, diff_tetha, diff_phi,
                     N_R, N_Tetha, N_Phi,
                     ind_w, ind_e,
                     theta_max=None):
    """
    This is the finite-volume approach for the integral, yet I did not use it since my approach for the whole problem
    was FD and not FV, hence, I thought combining FV and FD approach might feel weird and introduce inconsistency to the problem.
    Moreover, It was the case when I was checking the plots
    """
    i = N_R
    I = 0.0

    for j in range(N_Tetha):
        t_half = 0.5 * (tetha_points[j] + tetha_points[j+1])

        if (theta_max is not None) and (t_half > theta_max):
            continue

        Ajk = (r_sph**2) * sin(t_half) * diff_tetha * diff_phi

        for k in range(N_Phi):
            kp1 = 0 if (k == N_Phi-1) else (k+1)

            # corners: (j,k), (j+1,k), (j+1,k+1), (j,k+1)
            g00 = gfun(cwk_vec, cethk_vec, i, j,   k,   ind_w, ind_e)
            g10 = gfun(cwk_vec, cethk_vec, i, j+1, k,   ind_w, ind_e)
            g11 = gfun(cwk_vec, cethk_vec, i, j+1, kp1, ind_w, ind_e)
            g01 = gfun(cwk_vec, cethk_vec, i, j,   kp1, ind_w, ind_e)

            gI = 0.25 * (g00 + g10 + g11 + g01)
            I += gI * Ajk

    return I



#PDE system function
def PDE_sys2(x_init ,x_k, diff_t, r_points, diff_phi, diff_tetha,
             N_R_w, N_R_eth, N_Tetha, N_Phi, r_sph):
    """
    Whole PDE system which Broyden solver solve
    """

    # Total number of parameters for Ethanol and Water
    N_tot_w   = (N_R_w+1)*(N_Tetha+1)*(N_Phi+1)
    N_tot_eth = (N_R_eth+1)*(N_Tetha+1)*(N_Phi+1)
    off_eth = N_tot_w
    off_gas = N_tot_w + N_tot_eth

    # split previous step (x_init)
    x_init_w    = x_init[:N_tot_w]
    x_init_eth  = x_init[off_eth:off_gas]
    cn2g_prev   = x_init[off_gas]
    cwg_prev    = x_init[off_gas+1]
    cethg_prev  = x_init[off_gas+2]

    # split current unknowns (x_k)
    x_wk    = x_k[:N_tot_w]
    x_ethk  = x_k[off_eth:off_gas]
    cn2gk   = x_k[off_gas]
    cwgk    = x_k[off_gas+1]
    cethgk  = x_k[off_gas+2]

    # build liquid fields
    cw   = np.zeros((N_R_w+1,   N_Tetha+1, N_Phi+1), dtype=float)
    ceth = np.zeros((N_R_eth+1, N_Tetha+1, N_Phi+1), dtype=float)

    cwk   = np.zeros((N_R_w+1,   N_Tetha+1, N_Phi+1), dtype=float)
    cethk = np.zeros((N_R_eth+1, N_Tetha+1, N_Phi+1), dtype=float)

    # Turning vectors to matricies for water and ethanol
    for i in range(N_R_w+1):
        for j in range(N_Tetha+1):
            for k in range(N_Phi+1):
                l = indcies_dict_sph_w[(i, j, k)]
                cw[i, j, k]  = x_init_w[l]
                cwk[i, j, k] = x_wk[l]

    for i in range(N_R_eth+1):
        for j in range(N_Tetha+1):
            for k in range(N_Phi+1):
                l = indcies_dict_sph_eth[(i, j, k)]
                ceth[i, j, k]  = x_init_eth[l]
                cethk[i, j, k] = x_ethk[l]

    # residual matricies
    cw_res     = np.zeros_like(cwk)
    ceth_res   = np.zeros_like(cethk)
    cwg_res    = np.zeros((1,), dtype=float)
    cethg_res  = np.zeros((1,), dtype=float)
    cn2g_res   = np.zeros((1,), dtype=float)

    # residual arrays
    cw_res_vec   = np.zeros((N_tot_w,), dtype=float)
    ceth_res_vec = np.zeros((N_tot_eth,), dtype=float)

    # cwk_int_vec   = x_wk
    # cethk_int_vec = x_ethk

    def get_cw(i, j, k):
        l = indcies_dict_sph_w.get((i, j, k))
        return x_wk[l] if l is not None else 0.0

    def get_ceth(i, j, k):
        l = indcies_dict_sph_eth.get((i, j, k))
        return x_ethk[l] if l is not None else 0.0
    def surface_Iy_Aw_both_FD():
        """
        Same approach as Total liquid function has been used for the intergal(Non-uniform trapezodial integral)
        """
        dth  = tetha_points[1] - tetha_points[0]
        dphi = phi_points[1] - phi_points[0]

        Iw_y = 0.0
        Ieth_y = 0.0
        Aw = 0.0

        for j in range(N_Tetha + 1):
            th = tetha_points[j]
            if th > tetha_max:
                continue

            wj = 0.5 if (j == 0 or j == N_Tetha) else 1.0
            sinth = np.sin(th)

            # dA ring weight per node in phi:
            # dA = r^2 sinθ dθ dφ
            dA_node = (r_sph**2) * sinth * (wj * dth) * dphi

            for k in range(N_Phi):  
                Aw += dA_node

                # Equilibrium at the surface, additionally, checking the wetness too(for both water and ethanol)!
                if water_active:
                    ceth_at = cethk[N_R_eth, j, k] if ethanol_active else 0.0
                    xw = cwk[N_R_w, j, k] / (cwk[N_R_w, j, k] + ceth_at + 1e-20)
                    gamma_w = np.exp((0.9146 + 1.0736*xw) * (1 - xw)**2)
                    ystar_w = (Psat_w / P_use) * xw * gamma_w
                    Iw_y += ystar_w * dA_node

                if ethanol_active:
                    cw_at = cwk[N_R_w, j, k] if water_active else 0.0
                    xeth = cethk[N_R_eth, j, k] / (cw_at + cethk[N_R_eth, j, k] + 1e-20)
                    gamma_e = np.exp((1.4514 - 1.0736*xeth) * (1 - xeth)**2)
                    ystar_e = (Psat_eth / P_use) * xeth * gamma_e
                    Ieth_y += ystar_e * dA_node

        return Iw_y, Ieth_y, Aw


    def d1d2_nonuniform(c_im1, c_i, c_ip1, h1, h2):
        """
        In r-grid we have on-uniform spacing. Subcequently, we need to take first order
        and second order derivatives with non-uniform grid assumption. This function does the following
        computation:
        consider h1 = r_i - r_i-1 and h2 = r_i+1 - r_i
        d1 = (h_2^2(c_i-c_i-1)+h_1^2(c_i+1-c_i))/(h1h2(h1+h2))
        d2 = (2(h_2(c_i-1-c_i)+h_1(c_i+1-c_i)))/(h1h2(h1+h2))
        """
        denom = h1 * h2 * (h1 + h2)
        if denom == 0.0:
            return 0.0, 0.0
        d1 = (h2*h2 * (c_i - c_im1) + h1*h1 * (c_ip1 - c_i)) / denom
        d2 = 2.0 * (h2 * (c_im1 - c_i) + h1 * (c_ip1 - c_i)) / denom
        return d1, d2

    # Liquid internal nodes PDE
    if water_active:
        for i in range(1, N_R_w):
            r = r_points[i]
            h1 = r_points[i] - r_points[i-1]
            h2 = r_points[i+1] - r_points[i]
            for j in range(1, N_Tetha):
                th = tetha_points[j]
                sin_t = max(np.sin(th), 1e-8)
                cot_t = np.cos(th) / sin_t
                for k in range(N_Phi):
                    # These conditions forces 0 ~ N_phi in phi_direction(Peridioc BC)
                    if k == 0:
                        km1 = N_Phi-1; kp1 = 1
                    elif k == N_Phi-1:
                        km1 = N_Phi-2; kp1 = 0
                    else:
                        km1 = k-1;     kp1 = k+1

                    # Non-uniform derivatives in r-direction
                    dCdr, d2Cdr2 = d1d2_nonuniform(cwk[i-1,j,k], cwk[i,j,k], cwk[i+1,j,k], h1, h2)

                    # Uniform derivatives in tetha-direction
                    dCdth   = (cwk[i,j+1,k] - cwk[i,j-1,k]) / (2*diff_tetha)
                    d2Cdth2 = (cwk[i,j+1,k] - 2*cwk[i,j,k] + cwk[i,j-1,k]) / (diff_tetha**2)

                    # Uniform derivative in phi-direction
                    d2Cdph2 = (cwk[i,j,kp1] - 2*cwk[i,j,k] + cwk[i,j,km1])/(diff_phi**2)

                    # Laplacian calculation
                    Lap = (d2Cdr2 + (2.0/r)*dCdr
                           + (1.0/r**2)*(d2Cdth2 + cot_t*dCdth + (1.0/sin_t**2)*d2Cdph2))

                    # Building the water residual(Whole PDE in discretized form)
                    cw_res[i,j,k] = -cwk[i,j,k] + cw[i,j,k] + diff_t*D_eff_w*Lap

    # The logic is same for the ethanol
    if ethanol_active:
        for i in range(1, N_R_eth):
            r = r_points[i]
            h1 = r_points[i] - r_points[i-1]
            h2 = r_points[i+1] - r_points[i]
            for j in range(1, N_Tetha):
                th = tetha_points[j]
                sin_t = max(np.sin(th), 1e-8)
                cot_t = np.cos(th) / sin_t
                for k in range(N_Phi):
                    if k == 0:
                        km1 = N_Phi-1; kp1 = 1
                    elif k == N_Phi-1:
                        km1 = N_Phi-2; kp1 = 0
                    else:
                        km1 = k-1;     kp1 = k+1

                    dCdr, d2Cdr2 = d1d2_nonuniform(cethk[i-1,j,k], cethk[i,j,k], cethk[i+1,j,k], h1, h2)

                    dCdth   = (cethk[i,j+1,k] - cethk[i,j-1,k]) / (2*diff_tetha)
                    d2Cdth2 = (cethk[i,j+1,k] - 2*cethk[i,j,k] + cethk[i,j-1,k]) / (diff_tetha**2)

                    d2Cdph2 = (cethk[i,j,kp1] - 2*cethk[i,j,k] + cethk[i,j,km1])/(diff_phi**2)

                    Lap = (d2Cdr2 + (2.0/r)*dCdr
                           + (1.0/r**2)*(d2Cdth2 + cot_t*dCdth + (1.0/sin_t**2)*d2Cdph2))

                    ceth_res[i,j,k] = -cethk[i,j,k] + ceth[i,j,k] + diff_t*D_eff_eth*Lap

    # If water or ethanol gets dry set residuals equal to cw_k+1, ceth_k+1 = 0, 0
    if not water_active:
        cw_res[:,:,:] = cwk
    if not ethanol_active:
        ceth_res[:,:,:] = cethk


    # Gas phase volume  (mol/m^3)
    Vg = Vg_const

    # Total gas molar concentration from ideal gas law
    Cw_g   = max(cwgk,   0.0)
    Ceth_g = max(cethgk, 0.0)
    Cn2_g  = max(cn2gk,  0.0)

    Ct_bulk = max(cn2gk + cwgk + cethgk, 1e-30)   # mol/m^3
    P_use   = (Ct_bulk * R_gas * T) / 1e5         # bar

    # mole fractions
    yw_bulk   = Cw_g   / Ct_bulk
    yeth_bulk = Ceth_g / Ct_bulk
    yn2_bulk  = Cn2_g  / Ct_bulk


    # Mixture-averaged diffusivities in gas phase (m^2/s)
    D_wm   = D_mix_3(yeth_bulk, yn2_bulk,  DW_eth,  DW_n2)
    D_ethm = D_mix_3(yw_bulk,   yn2_bulk,  Deth_w,  Deth_n2)
    # Assumed gas hydrodynamics (Diffusion + convection case)
    mu_g   = 1.8e-5   # Pa*s  (typical gas viscosity near room T)
    u_g    = 0.0010     # m/s   (assumed effective circulation speed)
    Lc = 2.0 * r_sph  # diameter

    # density
    MW_mix = yw_bulk*(MWw/1000.0) + yeth_bulk*(MWeth/1000.0) + yn2_bulk*(MWn2/1000.0)
    rho_g = Ct_bulk * MW_mix

    # Computing kG for species based on experimental equations for sherwood number in sphere with convective and conduction mass transfer
    Re = rho_g * u_g * Lc / (mu_g + 1e-30)

    Sc_w   = mu_g / (rho_g * (D_wm   + 1e-30))
    Sc_eth = mu_g / (rho_g * (D_ethm + 1e-30))

    Sh_w   = 2.0 + 0.6*(Re**0.5)*(Sc_w**(1.0/3.0))
    Sh_eth = 2.0 + 0.6*(Re**0.5)*(Sc_eth**(1.0/3.0))

    ky_w   = Sh_w   * D_wm   / Lc
    ky_eth = Sh_eth * D_ethm / Lc

    # Molar transfer coeff k_G = k_y * Ctot  (mol/(m^2 s))
    kG_w   = ky_w   * Ct_bulk
    kG_eth = ky_eth * Ct_bulk

    ## Uncomment this and comment above block in order to consider mass-transfer just in diffusion case(sh ~ 2)
    # # Sherwood Sh=2 => k_y (m/s) (Diffusion only case)
    # ky_w   = D_wm   / r_sph
    # ky_eth = D_ethm / r_sph

    # # Molar transfer coeff k_G = k_y * Ctot  (mol/(m^2 s))
    # kG_w   = ky_w   * Ctot_var
    # kG_eth = ky_eth * Ctot_var



    # FD integrals over wet surface
    Iw_y, Ieth_y, A_wet = surface_Iy_Aw_both_FD()
 
    # gas residuals (mol/m^3)
    if water_active:
        cwg_res[0] = (Vg*(cwgk - cwg_prev)/diff_t
                      - kG_w*(Iw_y - yw_bulk*A_wet))
    else:
        cwg_res[0] = cwgk - cwg_prev

    if ethanol_active:
        cethg_res[0] = (Vg*(cethgk - cethg_prev)/diff_t
                        - kG_eth*(Ieth_y - yeth_bulk*A_wet))
   
    else:
        cethg_res[0] = cethgk - cethg_prev

    # total concentration of N2 is fixed
    cn2g_res[0] = cn2gk - Cn2_fixed



    # Boundary conditions
    dr_surf_w = (r_points[N_R_w] - r_points[N_R_w-1]) if N_R_w >= 1 else 0.0
    dr_surf_e = (r_points[N_R_eth] - r_points[N_R_eth-1]) if N_R_eth >= 1 else 0.0

    # At r = 0
    # we have rondC/rondr = 0
    # Excluding corners!
    for j in range(1, N_Tetha):
        for k in range(N_Phi):
            if water_active:
                cw_res[0, j, k] = cwk[0, j, k] - cwk[1, j, k]
            if ethanol_active:
                ceth_res[0, j, k] = cethk[0, j, k] - cethk[1, j, k]

    # At r = R_sponge
    # Robin(mixed) boundary condition
    for j in range(1, N_Tetha):
        for k in range(N_Phi):
            if water_active:
                ceth_at_w = get_ceth(N_R_w, j, k)
                xw = cwk[N_R_w, j, k] / (cwk[N_R_w, j, k] + ceth_at_w + 1e-20)
                ywg_star = Psat_w * xw / P_use * exp((0.9146 + 1.0736*xw)*(1-xw)**2)

                cw_res[N_R_w, j, k] = (cwk[N_R_w, j, k]
                                       + kG_w*dr_surf_w/D_eff_w*(ywg_star - yw_bulk)
                                       - cwk[N_R_w-1, j, k])

            if ethanol_active:
                cw_at_e = get_cw(N_R_eth, j, k)
                xeth = cethk[N_R_eth, j, k] / (cw_at_e + cethk[N_R_eth, j, k] + 1e-20)
                yethg_star = Psat_eth * xeth / P_use * exp((1.4514 - 1.0736*xeth)*(1-xeth)**2)

                ceth_res[N_R_eth, j, k] = (cethk[N_R_eth, j, k]
                                           + kG_eth*dr_surf_e/D_eff_eth*(yethg_star - yeth_bulk)
                                           - cethk[N_R_eth-1, j, k])

    # At theta = 0 and theta = pi
    # rondC/rondTetha at tetha =0 and tetha = tetha_max is equal to 0
    for i in range(1, N_R_w):
        for k in range(N_Phi):
            if water_active:
                cw_res[i, 0, k]        = cwk[i, 1, k] - cwk[i, 0, k]
                cw_res[i, N_Tetha, k]  = cwk[i, N_Tetha, k] - cwk[i, N_Tetha-1, k]
    for i in range(1, N_R_eth):
        for k in range(N_Phi):
            if ethanol_active:
                ceth_res[i, 0, k]       = cethk[i, 1, k] - cethk[i, 0, k]
                ceth_res[i, N_Tetha, k] = cethk[i, N_Tetha, k] - cethk[i, N_Tetha-1, k]

    # Periodic phi at k = N_Phi
    # C(N_phi) == C(0)
    for i in range(1, N_R_w):
        for j in range(1, N_Tetha):
            if water_active:
                cw_res[i, j, N_Phi] = cwk[i, j, N_Phi] - cwk[i, j, 0]
    for i in range(1, N_R_eth):
        for j in range(1, N_Tetha):
            if ethanol_active:
                ceth_res[i, j, N_Phi] = cethk[i, j, N_Phi] - cethk[i, j, 0]

    # Corner / weighted BCs
    # w1 and w2 should be unequal in order that Jacobian matrix will not be zero in rows!
    w1 = 0.5001
    w2 = 0.4999
    # BCS at r=0 and phi = N_phi(Last point of phi) at the same time for internal tethas
    for j in range(1, N_Tetha):
        if water_active:
            cw_res[0, j, N_Phi] = w1*(cwk[0, j, N_Phi]-cwk[1, j, N_Phi]) + w2*(cwk[0, j, N_Phi]-cwk[0, j, 0])
        if ethanol_active:
            ceth_res[0, j, N_Phi] = w1*(cethk[0, j, N_Phi]-cethk[1, j, N_Phi]) + w2*(cethk[0, j, N_Phi]-cethk[0, j, 0])
    # BCS at r=R_spng and phi = N_phi(Last point of phi) at the same time for internal tethas
        if water_active:
            ceth_at_w = get_ceth(N_R_w, j, N_Phi)
            xw = cwk[N_R_w, j, N_Phi] / (cwk[N_R_w, j, N_Phi] + ceth_at_w + 1e-20)
            ywg_star = Psat_w * xw / P_use * exp((0.9146 + 1.0736*xw)*(1-xw)**2)
            cw_res[N_R_w, j, N_Phi] = w1*(cwk[N_R_w, j, N_Phi] + kG_w*dr_surf_w/D_eff_w*(ywg_star - yw_bulk) - cwk[N_R_w-1, j, N_Phi]) \
                                      + w2*(cwk[N_R_w, j, N_Phi] - cwk[N_R_w, j, 0])

        if ethanol_active:
            cw_at_e = get_cw(N_R_eth, j, N_Phi)
            xeth = cethk[N_R_eth, j, N_Phi] / (cw_at_e + cethk[N_R_eth, j, N_Phi] + 1e-20)
            yethg_star = Psat_eth * xeth / P_use * exp((1.4514 - 1.0736*xeth)*(1-xeth)**2)
            ceth_res[N_R_eth, j, N_Phi] = w1*(cethk[N_R_eth, j, N_Phi] + kG_eth*dr_surf_e/D_eff_eth*(yethg_star - yeth_bulk) - cethk[N_R_eth-1, j, N_Phi]) \
                                          + w2*(cethk[N_R_eth, j, N_Phi] - cethk[N_R_eth, j, 0])

    # BCs at tetha = 0, phi = N_phi and tetha =N_Tetha, phi = N_phi at interanl rs
    for i in range(1, N_R_w):
        if water_active:
            cw_res[i, 0, N_Phi] = w1*(cwk[i, 1, N_Phi]-cwk[i, 0, N_Phi]) + w2*(cwk[i, 0, N_Phi]-cwk[i, 0, 0])
            cw_res[i, N_Tetha, N_Phi] = w1*(cwk[i, N_Tetha, N_Phi]-cwk[i, N_Tetha-1, N_Phi]) + w2*(cwk[i, N_Tetha, N_Phi]-cwk[i, N_Tetha, 0])

    for i in range(1, N_R_eth):
        if ethanol_active:
            ceth_res[i, 0, N_Phi] = w1*(cethk[i, 1, N_Phi]-cethk[i, 0, N_Phi]) + w2*(cethk[i, 0, N_Phi]-cethk[i, 0, 0])
            ceth_res[i, N_Tetha, N_Phi] = w1*(cethk[i, N_Tetha, N_Phi]-cethk[i, N_Tetha-1, N_Phi]) + w2*(cethk[i, N_Tetha, N_Phi]-cethk[i, N_Tetha, 0])
    # BCs at tetha = 0, r = 0 and tetha =N_Tetha, r = N_r and tetha = 0 and r = N_r at interanl phis
    for k in range(N_Phi):
        if water_active:
            cw_res[0, 0, k] = w1*(cwk[0, 0, k]-cwk[1, 0, k]) + w2*(cwk[0, 1, k]-cwk[0, 0, k])
            cw_res[0, N_Tetha, k] = w1*(cwk[0, N_Tetha, k]-cwk[1, N_Tetha, k]) + w2*(cwk[0, N_Tetha, k]-cwk[0, N_Tetha-1, k])

        if ethanol_active:
            ceth_res[0, 0, k] = w1*(cethk[0, 0, k]-cethk[1, 0, k]) + w2*(cethk[0, 1, k]-cethk[0, 0, k])
            ceth_res[0, N_Tetha, k] = w1*(cethk[0, N_Tetha, k]-cethk[1, N_Tetha, k]) + w2*(cethk[0, N_Tetha, k]-cethk[0, N_Tetha-1, k])

        if water_active:
            ceth_at_w = get_ceth(N_R_w, 0, k)
            xw = cwk[N_R_w, 0, k]/(cwk[N_R_w, 0, k]+ceth_at_w+1e-20)
            ywg_star = Psat_w*xw/P_use*exp((0.9146+1.0736*xw)*(1-xw)**2)
            cw_res[N_R_w, 0, k] = w1*(cwk[N_R_w, 0, k] + kG_w*dr_surf_w/D_eff_w*(ywg_star - yw_bulk) - cwk[N_R_w-1, 0, k]) \
                                  + w2*(cwk[N_R_w, 1, k] - cwk[N_R_w, 0, k])

        if ethanol_active:
            cw_at_e = get_cw(N_R_eth, 0, k)
            xeth = cethk[N_R_eth, 0, k]/(cw_at_e + cethk[N_R_eth, 0, k] + 1e-20)
            yethg_star = Psat_eth*xeth/P_use*exp((1.4514-1.0736*xeth)*(1-xeth)**2)
            ceth_res[N_R_eth, 0, k] = w1*(cethk[N_R_eth, 0, k] + kG_eth*dr_surf_e/D_eff_eth*(yethg_star - yeth_bulk) - cethk[N_R_eth-1, 0, k]) \
                                      + w2*(cethk[N_R_eth, 1, k] - cethk[N_R_eth, 0, k])

        if water_active:
            ceth_at_w = get_ceth(N_R_w, N_Tetha, k)
            xw = cwk[N_R_w, N_Tetha, k]/(cwk[N_R_w, N_Tetha, k]+ceth_at_w+1e-20)
            ywg_star = Psat_w*xw/P_use*exp((0.9146+1.0736*xw)*(1-xw)**2)
            cw_res[N_R_w, N_Tetha, k] = w1*(cwk[N_R_w, N_Tetha, k] + kG_w*dr_surf_w/D_eff_w*(ywg_star - yw_bulk) - cwk[N_R_w-1, N_Tetha, k]) \
                                        + w2*(cwk[N_R_w, N_Tetha, k] - cwk[N_R_w, N_Tetha-1, k])

        if ethanol_active:
            cw_at_e = get_cw(N_R_eth, N_Tetha, k)
            xeth = cethk[N_R_eth, N_Tetha, k]/(cw_at_e + cethk[N_R_eth, N_Tetha, k] + 1e-20)
            yethg_star = Psat_eth*xeth/P_use*exp((1.4514-1.0736*xeth)*(1-xeth)**2)
            ceth_res[N_R_eth, N_Tetha, k] = w1*(cethk[N_R_eth, N_Tetha, k] + kG_eth*dr_surf_e/D_eff_eth*(yethg_star - yeth_bulk) - cethk[N_R_eth-1, N_Tetha, k]) \
                                            + w2*(cethk[N_R_eth, N_Tetha, k] - cethk[N_R_eth, N_Tetha-1, k])

    # Three-BC corners
    # More weight on r-direction BCs since it matters more for our plots
    w1 = 0.5
    w2 = 0.25
    w3 = 0.25

    # BCs at r=0, tetha = 0, phi =0
    if water_active:
        cw_res[0, 0, N_Phi] = w1*(cwk[0,0,N_Phi]-cwk[1,0,N_Phi]) + w2*(cwk[0,1,N_Phi]-cwk[0,0,N_Phi]) + w3*(cwk[0,0,N_Phi]-cwk[0,0,0])
    if ethanol_active:
        ceth_res[0, 0, N_Phi] = w1*(cethk[0,0,N_Phi]-cethk[1,0,N_Phi]) + w2*(cethk[0,1,N_Phi]-cethk[0,0,N_Phi]) + w3*(cethk[0,0,N_Phi]-cethk[0,0,0])
    # BCs at r=0, tetha = N_Tetha, phi = N_phi
    if water_active:
        cw_res[0, N_Tetha, N_Phi] = w1*(cwk[0,N_Tetha,N_Phi]-cwk[1,N_Tetha,N_Phi]) + w2*(cwk[0,N_Tetha,N_Phi]-cwk[0,N_Tetha-1,N_Phi]) + w3*(cwk[0,N_Tetha,N_Phi]-cwk[0,N_Tetha,0])
    if ethanol_active:
        ceth_res[0, N_Tetha, N_Phi] = w1*(cethk[0,N_Tetha,N_Phi]-cethk[1,N_Tetha,N_Phi]) + w2*(cethk[0,N_Tetha,N_Phi]-cethk[0,N_Tetha-1,N_Phi]) + w3*(cethk[0,N_Tetha,N_Phi]-cethk[0,N_Tetha,0])
    # BCs at r=N_r, tetha = 0, phi = N_phi
    if water_active:
        ceth_at_w = get_ceth(N_R_w, 0, N_Phi)
        xw = cwk[N_R_w, 0, N_Phi]/(cwk[N_R_w, 0, N_Phi]+ceth_at_w+1e-20)
        ywg_star = Psat_w*xw/P_use*exp((0.9146+1.0736*xw)*(1-xw)**2)
        cw_res[N_R_w, 0, N_Phi] = w1*(cwk[N_R_w,0,N_Phi] + kG_w*dr_surf_w/D_eff_w*(ywg_star - yw_bulk) - cwk[N_R_w-1,0,N_Phi]) \
                                  + w2*(cwk[N_R_w,1,N_Phi] - cwk[N_R_w,0,N_Phi]) \
                                  + w3*(cwk[N_R_w,0,N_Phi] - cwk[N_R_w,0,0])

    if ethanol_active:
        cw_at_e = get_cw(N_R_eth, 0, N_Phi)
        xeth = cethk[N_R_eth, 0, N_Phi]/(cw_at_e + cethk[N_R_eth, 0, N_Phi]+1e-20)
        yethg_star = Psat_eth*xeth/P_use*exp((1.4514-1.0736*xeth)*(1-xeth)**2)
        ceth_res[N_R_eth, 0, N_Phi] = w1*(cethk[N_R_eth,0,N_Phi] + kG_eth*dr_surf_e/D_eff_eth*(yethg_star - yeth_bulk) - cethk[N_R_eth-1,0,N_Phi]) \
                                      + w2*(cethk[N_R_eth,1,N_Phi] - cethk[N_R_eth,0,N_Phi]) \
                                      + w3*(cethk[N_R_eth,0,N_Phi] - cethk[N_R_eth,0,0])
    # BCs at r=N_r, tetha = N_tetha, phi = N_phi
    if water_active:
        ceth_at_w = get_ceth(N_R_w, N_Tetha, N_Phi)
        xw = cwk[N_R_w, N_Tetha, N_Phi]/(cwk[N_R_w, N_Tetha, N_Phi]+ceth_at_w+1e-20)
        ywg_star = Psat_w*xw/P_use*exp((0.9146+1.0736*xw)*(1-xw)**2)
        cw_res[N_R_w, N_Tetha, N_Phi] = w1*(cwk[N_R_w,N_Tetha,N_Phi] + kG_w*dr_surf_w/D_eff_w*(ywg_star - yw_bulk) - cwk[N_R_w-1,N_Tetha,N_Phi]) \
                                        + w2*(cwk[N_R_w,N_Tetha,N_Phi] - cwk[N_R_w,N_Tetha-1,N_Phi]) \
                                        + w3*(cwk[N_R_w,N_Tetha,N_Phi] - cwk[N_R_w,N_Tetha,0])

    if ethanol_active:
        cw_at_e = get_cw(N_R_eth, N_Tetha, N_Phi)
        xeth = cethk[N_R_eth, N_Tetha, N_Phi]/(cw_at_e + cethk[N_R_eth, N_Tetha, N_Phi]+1e-20)
        yethg_star = Psat_eth*xeth/P_use*exp((1.4514-1.0736*xeth)*(1-xeth)**2)
        ceth_res[N_R_eth, N_Tetha, N_Phi] = w1*(cethk[N_R_eth,N_Tetha,N_Phi] + kG_eth*dr_surf_e/D_eff_eth*(yethg_star - yeth_bulk) - cethk[N_R_eth-1,N_Tetha,N_Phi]) \
                                            + w2*(cethk[N_R_eth,N_Tetha,N_Phi] - cethk[N_R_eth,N_Tetha-1,N_Phi]) \
                                            + w3*(cethk[N_R_eth,N_Tetha,N_Phi] - cethk[N_R_eth,N_Tetha,0])

    # pack residual in vectors
    for i in range(N_R_w+1):
        for j in range(N_Tetha+1):
            for k in range(N_Phi+1):
                l = indcies_dict_sph_w[(i, j, k)]
                cw_res_vec[l] = cw_res[i, j, k]

    for i in range(N_R_eth+1):
        for j in range(N_Tetha+1):
            for k in range(N_Phi+1):
                l = indcies_dict_sph_eth[(i, j, k)]
                ceth_res_vec[l] = ceth_res[i, j, k]

    # Return residuals as vectors
    f_tot = np.concatenate((cw_res_vec, ceth_res_vec, cwg_res, cethg_res, cn2g_res))
    return f_tot


def make_wrapped_PDE(x_old_fixed, dt_local):
    """
    This function does 2 things:
    1-Freezes the previous time level state x_k-1(last accepted solution)
    2-Returns a function with only unknowns x_k so non-linear solver can iterate on it
    """
    # freeze previous accepted state for this step
    x_init = np.array(x_old_fixed, dtype=float, copy=True)

    # Define the function the solver will call many times
    def wrapped_PDE(x_new):
        # x_new is what Broyden/Householder changes
        xk = init_xk(x_new, N_tot_w, N_tot_eth)
        return PDE_sys2(
            x_init, xk,
            dt_local, r_points, diff_phi, diff_tetha,
            N_R_w, N_R_eth, N_Tetha, N_Phi, r_sph
        )
    return wrapped_PDE



def startup_build_coloring_from_numeric_J(f, x0, rel_step=1e-7, nz_tol=1e-12):
    """
    ONE-TIME startup:
      1-compute numeric Jacobian Jref at x0
      2- build rows_by_col from true nonzero pattern of Jref
      3-greedy color to build groups
      4-force last-3 columns (gas vars) to be singleton colors
    Returns: rows_by_col, groups
    """

    #Evaluate F(x0)
    x0 = np.asarray(x0, float)
    F0 = np.asarray(f(x0), float)

    # full numeric Jacobian ONCE
    Jref = num_jacobe_fast(f, x0, F0, rel_step)

    # sparsity from Jref
    # Checking which enteries in jacobian matrix is zero or not in a bollean matrix
    # nz_tol is 1e-12
    S = np.abs(Jref) > nz_tol
    # Which residual depend on variable xc, find when it's true in each row on c column
    rows_by_col = [np.where(S[:, c])[0].astype(int) for c in range(Jref.shape[1])]

    # greedy coloring
    groups, ncolors = build_groups_greedy(rows_by_col)
    groups = [np.asarray(g, dtype=int).ravel() for g in groups]

    # force last 3 variables singleton (your gas vars)
    n = x0.size
    gas_cols = np.array([n-3, n-2, n-1], dtype=int)

    new_groups = []
    for g in groups:
        # Removing gas columns from old groups
        g = g[~np.isin(g, gas_cols)]
        if g.size:
            new_groups.append(g)
    # Adding gas columns seperately as a group!
    for gc in gas_cols:
        new_groups.append(np.array([gc], dtype=int))
    groups = new_groups

    print("Startup: colors =", len(groups), " n =", x0.size, " m =", F0.size)
    return rows_by_col, groups

def remap_state(x_old_in, old_N_tot_w, old_N_tot_eth, old_dict_w, old_dict_eth,
                new_N_tot_w, new_N_tot_eth, new_dict_w, new_dict_eth):
    """
    This function carry the solution to a new grid whenever N_R for ethanol/water shrinks!
    """
    # Create a new zero state with new sizes
    new_x = np.zeros((new_N_tot_w + new_N_tot_eth + 3,), dtype=float)
    # Checking if node still exist after shrinkage
    # If exist copy its value in new grid
    # Else do nothing
    for key, new_l in new_dict_w.items():
        old_l = old_dict_w.get(key)
        if old_l is not None:
            new_x[new_l] = x_old_in[old_l]
    for key, new_l in new_dict_eth.items():
        old_l = old_dict_eth.get(key)
        if old_l is not None:
            new_x[new_N_tot_w + new_l] = x_old_in[old_N_tot_w + old_l]
    old_gas = old_N_tot_w + old_N_tot_eth
    new_gas = new_N_tot_w + new_N_tot_eth
    new_x[new_gas:new_gas+3] = x_old_in[old_gas:old_gas+3]
    return new_x


def setup_solver_state(x_old_in=None, old_state=None):
    """
    Setting the solver attributes like groups and colors(For the jacobian)
    and also update intial condition If the grid shrinks!
    """
    global N_tot_w, N_tot_eth, m_sph_w, m_sph_eth
    global indcies_dict_sph_w, indcies_dict_sph_rev_w
    global indcies_dict_sph_eth, indcies_dict_sph_rev_eth
    global x_old, rows_by_col, groups

    if old_state is None:
        old_state = {
            "N_tot_w": N_tot_w,
            "N_tot_eth": N_tot_eth,
            "dict_w": indcies_dict_sph_w,
            "dict_eth": indcies_dict_sph_eth,
        }

    m_sph_w, N_tot_w, indcies_dict_sph_w, indcies_dict_sph_rev_w, \
    m_sph_eth, N_tot_eth, indcies_dict_sph_eth, indcies_dict_sph_rev_eth = rebuild_grid_dependent_state(N_R_w, N_R_eth)

    if x_old_in is None:
        x_old = init(N_tot_w, N_tot_eth)
    else:
        x_old = remap_state(
            x_old_in,
            old_state["N_tot_w"], old_state["N_tot_eth"],
            old_state["dict_w"], old_state["dict_eth"],
            N_tot_w, N_tot_eth,
            indcies_dict_sph_w, indcies_dict_sph_eth
        )

    wrapped0 = make_wrapped_PDE(x_old, dt)
    rows_by_col, groups = startup_build_coloring_from_numeric_J(
        wrapped0, x_old, rel_step=1e-7, nz_tol=1e-12
    )
    import Broyden8
    Broyden8.rows_by_col = rows_by_col
    Broyden8.groups = groups


setup_solver_state()







def clamp_surface_nonpositive_at_min_radius(x_new):
    """
    Checking the solution vector, If it is negative clamped its value
    and return the change flag!
    """
    x_w_k = x_new[:N_tot_w]
    x_eth_k = x_new[N_tot_w:N_tot_w+N_tot_eth]
    changed = False
    for j in range(N_Tetha + 1):
        for k in range(N_Phi + 1):
            if water_active:
                l = indcies_dict_sph_w[(N_R_w, j, k)]
                if x_w_k[l] <= 0.0:
                    x_w_k[l] = 0.0
                    changed = True
            if ethanol_active:
                l = indcies_dict_sph_eth[(N_R_eth, j, k)]
                if x_eth_k[l] <= 0.0:
                    x_eth_k[l] = 0.0
                    changed = True
    water_dried = np.max(x_w_k) <= 0.0
    ethanol_dried = np.max(x_eth_k) <= 0.0
    return changed, water_dried, ethanol_dried


def is_steady_state(x_new, x_old, tol):
    """
    Checking if we have reached to st.st or not based on tol
    """
    return np.max(np.abs(x_new - x_old)) <= tol


def liquid_dried(x_new, species):
    """
    Checks the complete dryness condition. If some nodes still has liquid it returns false!
    """
    if species == "water":
        x_k = x_new[:N_tot_w]
    else:
        x_k = x_new[N_tot_w:N_tot_w+N_tot_eth]
    return np.max(x_k) <= 0.0


def log_state(x_vec, label):
    """
    logs the min/max concentration of water and ethanol, moreover, logs min concentration at the surface, too.
    """
    x_w = x_vec[:N_tot_w]
    x_e = x_vec[N_tot_w:N_tot_w+N_tot_eth]
    w_min = float(np.min(x_w)) if x_w.size else float("nan")
    w_max = float(np.max(x_w)) if x_w.size else float("nan")
    e_min = float(np.min(x_e)) if x_e.size else float("nan")
    e_max = float(np.max(x_e)) if x_e.size else float("nan")
    # surface mins
    w_surf_min = float("nan")
    e_surf_min = float("nan")
    if x_w.size:
        w_surf_vals = []
        for j in range(N_Tetha + 1):
            for k in range(N_Phi + 1):
                l = indcies_dict_sph_w[(N_R_w, j, k)]
                w_surf_vals.append(x_w[l])
        w_surf_min = float(np.min(w_surf_vals))
    if x_e.size:
        e_surf_vals = []
        for j in range(N_Tetha + 1):
            for k in range(N_Phi + 1):
                l = indcies_dict_sph_eth[(N_R_eth, j, k)]
                e_surf_vals.append(x_e[l])
        e_surf_min = float(np.min(e_surf_vals))

    print(f"[DBG:{label}] n={n} N_R_w={N_R_w} N_R_eth={N_R_eth} "
          f"w[min,max]=({w_min:.3e},{w_max:.3e}) e[min,max]=({e_min:.3e},{e_max:.3e}) "
          f"w_surf_min={w_surf_min:.3e} e_surf_min={e_surf_min:.3e}")



# storage (ALL time steps)
n = 0
steady_tol = 1e-8
water_dried_reported = False
ethanol_dried_reported = False
failed_at_eth_min = False

ts = []
cw_surf_ts = []     # mean over (theta,phi) at i=N_R_w
ceth_surf_ts = []   # mean over (theta,phi) at i=N_R_eth
cwg_ts = []
cethg_ts = []
cn2g_ts = []
P_ts = []
# store full surface fields for ALL times
#   cw_NR_all   -> (Nt_saved, N_Tetha+1, N_Phi+1)
#   ceth_NR_all -> (Nt_saved, N_Tetha+1, N_Phi+1)
cw_NR_all = []
ceth_NR_all = []

# Radial mean profiles over time 
cw_rmean_ts   = []   # each entry length N_R_BASE+1
ceth_rmean_ts = []   # each entry length N_R_BASE+1

nW_liq_ts, nW_gas_ts, nW_tot_ts = [], [], []
nE_liq_ts, nE_gas_ts, nE_tot_ts = [], [], []
# initial totals (from the initial state x_old at t=0)
nW_liq0 = total_moles_liquid_species_FD(
    x_old, "water",
    N_R_w, N_R_eth, N_Tetha, N_Phi,
    r_points, tetha_points, phi_points,
    indcies_dict_sph_w, indcies_dict_sph_eth,
    N_tot_w, N_tot_eth,
    theta_max=tetha_max
)
nW_gas0 = total_moles_gas_species(x_old, "water", Vg_const, N_tot_w, N_tot_eth)
nW_tot0 = nW_liq0 + nW_gas0

nE_liq0 = total_moles_liquid_species_FD(
    x_old, "ethanol",
    N_R_w, N_R_eth, N_Tetha, N_Phi,
    r_points, tetha_points, phi_points,
    indcies_dict_sph_w, indcies_dict_sph_eth,
    N_tot_w, N_tot_eth,
    theta_max=tetha_max
)
nE_gas0 = total_moles_gas_species(x_old, "ethanol", Vg_const, N_tot_w, N_tot_eth)
nE_tot0 = nE_liq0 + nE_gas0


# seed gas concentrations + pressure at t=0
gas_off0 = N_tot_w + N_tot_eth
cn2_0 = float(x_old[gas_off0])
cwG_0 = float(x_old[gas_off0 + 1])
ceG_0 = float(x_old[gas_off0 + 2])

cn2g_ts.append(cn2_0)
cwg_ts.append(cwG_0)
cethg_ts.append(ceG_0)

Ct0 = max(cn2_0 + cwG_0 + ceG_0, 1e-30)
P_ts.append((Ct0 * R_gas * T) / 1e5)
            
n = 0  # step counter 

# Solver loop
while t_now < t_end:

    # don’t step beyond final time
    dt = min(dt, t_end - t_now)

    x_prev = np.array(x_old, copy=True)   # <store last accepted state

    wrapped = make_wrapped_PDE(x_old, dt)
    x_guess = np.array(x_old, copy=True)


    x_new, Fk, ok, it = householder_like_lu(
        wrapped, x_guess,
        eps=1e-8, max_outer_num=50, max_inner_num=20,
        rel_step=1e-6, verbose=False, N_tot=(N_tot_w + N_tot_eth)
    )

    # STEP REJECTION ON SOLVER FAIL
    if not ok:
        log_state(x_new, "solver_fail")
        print(f"[DBG:solver_fail] it={it} ||F||={float(L2norm(Fk)):.3e}")

        # If either water  gets dry dropping water and continue with the other one+gases(This is the sepcial case that does not happend in the first place)
        if water_active and N_R_w <= 1:
            print("Solver failed with N_R_w<=1; dropping water equations and continuing with ethanol/gas only.")
            old_state = {
                "N_tot_w": N_tot_w,
                "N_tot_eth": N_tot_eth,
                "dict_w": indcies_dict_sph_w,
                "dict_eth": indcies_dict_sph_eth,
            }
            water_active = False
            N_R_w = -1
            setup_solver_state(x_old, old_state=old_state)
            # reject time step AND shrink dt
            dt = max(dt * shrink_reject, dt_min)
            continue

        if ethanol_active and N_R_eth <= 1:
            failed_at_eth_min = True
            print("Solver failed with N_R_eth<=1; stopping and plotting time series.")
            break

        # reject time step AND shrink dt on failed solve
        dt = max(dt * shrink_reject, dt_min)
        print(f"[ADAPT] reject: solver failed dt={dt:.3e}")
        if dt <= dt_min + 1e-15:
            print("[ADAPT] dt hit dt_min; stopping.")
            break
        continue

    # check surface negatives
    water_surface_nonpos = False
    ethanol_surface_nonpos = False

    if water_active:
        for j in range(N_Tetha + 1):
            for k in range(N_Phi + 1):
                l = indcies_dict_sph_w[(N_R_w, j, k)]
                if x_new[l] < 0.0:
                    water_surface_nonpos = True
                    break
            if water_surface_nonpos:
                break

    if ethanol_active:
        for j in range(N_Tetha + 1):
            for k in range(N_Phi + 1):
                l = indcies_dict_sph_eth[(N_R_eth, j, k)]
                if x_new[N_tot_w + l] < 0.0:
                    ethanol_surface_nonpos = True
                    break
            if ethanol_surface_nonpos:
                break
    # If we encounter to non-positivity(We do not in the problem!), first set retry_step flag flase
    if water_surface_nonpos or ethanol_surface_nonpos:
        old_state = {
            "N_tot_w": N_tot_w,
            "N_tot_eth": N_tot_eth,
            "dict_w": indcies_dict_sph_w,
            "dict_eth": indcies_dict_sph_eth,
        }
        retry_step = False

        # Check if we still have water in the sponge
        if water_surface_nonpos and water_active:
            for j in range(N_Tetha + 1):
                for k in range(N_Phi + 1):
                    l = indcies_dict_sph_w[(N_R_w, j, k)]
                    if x_new[l] <= 0.0:
                        x_new[l] = 0.0
            # If N_R_w keeps decreasing till the first node we report water has completely dreid off
            if N_R_w <= 1:
                changed, water_dried, _ = clamp_surface_nonpositive_at_min_radius(x_new)
                if changed:
                    print("Water surface nonpositive at N_R_w=1; clamped to 0 and continuing.")
                log_state(x_new, "water_surface_nonpos_at_min")
                if water_dried and not water_dried_reported:
                    print("Water has dried up completely.")
                    water_dried_reported = True
            # Else shrink the N_R_w and solve the problem again(This is actually r_interface for the water)
            else:
                N_R_w -= 1
                print(f"Water surface nonpositive; reducing N_R_w to {N_R_w} and retrying step {n}.")
                retry_step = True

        # Same for ethanol
        if ethanol_surface_nonpos and ethanol_active:
            for j in range(N_Tetha + 1):
                for k in range(N_Phi + 1):
                    l = indcies_dict_sph_eth[(N_R_eth, j, k)]
                    idx = N_tot_w + l
                    if x_new[idx] <= 0.0:
                        x_new[idx] = 0.0
            if N_R_eth <= 1:
                changed, _, ethanol_dried = clamp_surface_nonpositive_at_min_radius(x_new)
                if changed:
                    print("Ethanol surface nonpositive at N_R_eth=1; clamped to 0 and continuing.")
                log_state(x_new, "eth_surface_nonpos_at_min")
                if ethanol_dried and not ethanol_dried_reported:
                    print("Ethanol has dried up completely.")
                    ethanol_dried_reported = True
            else:
                N_R_eth -= 1
                print(f"Ethanol surface nonpositive; reducing N_R_eth to {N_R_eth} and retrying step {n}.")
                retry_step = True

        # If retry is true solve the problem again
        if retry_step:
            setup_solver_state(x_old, old_state=old_state)
            dt = max(dt * shrink_reject, dt_min)
            continue


    # ACCEPT STEP (advance time)
    t_now += dt

    # steady state / drying checks 
    if is_steady_state(x_new, x_old, steady_tol):
        print(f"Steady state reached at time step {n}; stopping.")
        x_old = x_new
        steady_break = True
    else:
        steady_break = False

    if water_active and liquid_dried(x_new, "water"):
        water_active = False
        x_new[:N_tot_w] = 0.0
        if not water_dried_reported:
            print("Water has dried up completely.")
            water_dried_reported = True

    if ethanol_active and liquid_dried(x_new, "ethanol"):
        ethanol_active = False
        x_new[N_tot_w:N_tot_w+N_tot_eth] = 0.0
        if not ethanol_dried_reported:
            print("Ethanol has dried up completely.")
            ethanol_dried_reported = True

    # now we finalize the state
    x_old = x_new
    print(f"Convergence were successful at time step {n}, moving to next step.")

    # 
    # RECORD (ALL TIME STEPS)
    # 
    ts.append(float(t_now))   

    # gas
    gas_off = N_tot_w + N_tot_eth
    cn2 = float(x_new[gas_off])        # mol/m^3
    cwG = float(x_new[gas_off + 1])    # mol/m^3
    ceG = float(x_new[gas_off + 2])    # mol/m^3

    cn2g_ts.append(cn2)
    cwg_ts.append(cwG)
    cethg_ts.append(ceG)

    # pressure from ideal gas law (bar)
    Ct = max(cn2 + cwG + ceG, 1e-30)
    P_use = (Ct * R_gas * T) / 1e5
    P_ts.append(P_use)

    Pmax = 3 * 1.01325
    if P_use > Pmax:
        print(f"[WARN] P_use={P_use:.4f} bar exceeds 3 atm ({Pmax:.4f} bar) at step n={n}, t={ts[-1]:.1f}s")

    # MASS CONSERVATION AUDIT 
    # Appending the total moles of liquid and gas at each time step!
    nW_liq = total_moles_liquid_species_FD(
    x_new, "water",
    N_R_w, N_R_eth, N_Tetha, N_Phi,
    r_points, tetha_points, phi_points,
    indcies_dict_sph_w, indcies_dict_sph_eth,
    N_tot_w, N_tot_eth,
    theta_max=tetha_max)

    nW_gas = total_moles_gas_species(x_new, "water", Vg_const, N_tot_w, N_tot_eth)
    nW_tot = nW_liq + nW_gas

    nE_liq = total_moles_liquid_species_FD(
    x_new, "ethanol",
    N_R_w, N_R_eth, N_Tetha, N_Phi,
    r_points, tetha_points, phi_points,
    indcies_dict_sph_w, indcies_dict_sph_eth,
    N_tot_w, N_tot_eth,
    theta_max=tetha_max)

    nE_gas = total_moles_gas_species(x_new, "ethanol", Vg_const, N_tot_w, N_tot_eth)
    nE_tot = nE_liq + nE_gas

    nW_liq_ts.append(nW_liq); nW_gas_ts.append(nW_gas); nW_tot_ts.append(nW_tot)
    nE_liq_ts.append(nE_liq); nE_gas_ts.append(nE_gas); nE_tot_ts.append(nE_tot)

    # RADIAL MEAN LIQUID PROFILES 
    cw_rmean   = np.full((N_R_BASE+1,), np.nan, dtype=float)
    ceth_rmean = np.full((N_R_BASE+1,), np.nan, dtype=float)

    # Checking the water is present and active
    if N_R_w >= 0 and N_tot_w > 0:
        if water_active:
            for i in range(N_R_w + 1):
                vals = []
                for j in range(N_Tetha + 1):
                    for k in range(N_Phi + 1):
                        l = indcies_dict_sph_w[(i, j, k)]
                        vals.append(x_new[l])
                cw_rmean[i] = float(np.mean(vals))
        # If water is not active then set the mean concentration 0
        else:
            cw_rmean[:N_R_w+1] = 0.0

    if N_R_eth >= 0 and N_tot_eth > 0:
        if ethanol_active:
            for i in range(N_R_eth + 1):
                vals = []
                for j in range(N_Tetha + 1):
                    for k in range(N_Phi + 1):
                        l = indcies_dict_sph_eth[(i, j, k)]
                        vals.append(x_new[N_tot_w + l])
                ceth_rmean[i] = float(np.mean(vals))
        else:
            ceth_rmean[:N_R_eth+1] = 0.0

    cw_rmean_ts.append(cw_rmean)
    ceth_rmean_ts.append(ceth_rmean)

        # STORE SURFACE FIELDS (i = N_R_w / N_R_eth)
    if N_R_w >= 0 and N_tot_w > 0:
        cw_NR = np.zeros((N_Tetha+1, N_Phi+1), dtype=float)
        for j in range(N_Tetha+1):
            for k in range(N_Phi+1):
                l = indcies_dict_sph_w[(N_R_w, j, k)]
                cw_NR[j, k] = x_new[l]
        cw_NR_all.append(cw_NR)

    if N_R_eth >= 0 and N_tot_eth > 0:
        ceth_NR = np.zeros((N_Tetha+1, N_Phi+1), dtype=float)
        for j in range(N_Tetha+1):
            for k in range(N_Phi+1):
                l = indcies_dict_sph_eth[(N_R_eth, j, k)]
                ceth_NR[j, k] = x_new[N_tot_w + l]
        ceth_NR_all.append(ceth_NR)

    # DEBUG: angular variation check on surface (theta/phi)
    if DBG_ANG_EVERY and (n % DBG_ANG_EVERY == 0):
        if N_R_w >= 0 and N_tot_w > 0:
            w_min = float(np.min(cw_NR))
            w_max = float(np.max(cw_NR))
            print(f"[DBG:ang] n={n} t={t_now:.3e} cw surface Tetha/Phi range = {w_min:.3e} .. {w_max:.3e}")
        if N_R_eth >= 0 and N_tot_eth > 0:
            e_min = float(np.min(ceth_NR))
            e_max = float(np.max(ceth_NR))
            print(f"[DBG:ang] n={n} t={t_now:.3e} ceth surface Tetha/Phi range = {e_min:.3e} .. {e_max:.3e}")
    
    #  STORE SURFACE MEANS 
    if N_R_w >= 0 and N_tot_w > 0:
        cw_surf_ts.append(float(np.mean(cw_NR)))   # uses cw_NR from above
    else:
        cw_surf_ts.append(np.nan)

    if N_R_eth >= 0 and N_tot_eth > 0:
        ceth_surf_ts.append(float(np.mean(ceth_NR)))  # uses ceth_NR from above
    else:
        ceth_surf_ts.append(np.nan)


    # surface storage blocks 

    if steady_break:
        break
    
    # Adaptive time block
    # Checking how much dx has changed
    dx = x_new - x_prev
    # sc maximum available tolerance
    sc = atol + rtol * np.maximum(np.abs(x_new), np.abs(x_prev))
    # Scale the error
    q  = dx / (sc + 1e-30)

    # RMS-style error
    eta = float(np.sqrt(np.mean(q*q)))

    # Hysteresis band(Decides if step was easy for the solver or hard)
    eta_grow   = 0.6     # easy step
    eta_shrink = 1.2     # hard step

    expo = 1.0/(order + 1.0)

    # Easy step because x_new did not change much and solver iterated steps was not that many
    if eta < eta_grow and it <= it_target:
        fac = safety * (1.0/max(eta, 1e-30))**expo
        fac = min(3, fac)         # allow faster growth
    # Hard step since x_new has changed a lot and solver steps were a lot, too!
    elif eta > eta_shrink or it >= it_hard:
        fac = safety * (1.0/max(eta, 1e-30))**expo
        fac = max(0.6, fac)         # don't murder dt
    # Middle step(It is either solver value has changed so much, yet the iterated steps were not as much or iteration steps were many)
    else:
        fac = 1.0                   

    # If dt is already large, cap the growth factor to avoid huge jumps
    # It is because at large values since we reach towards st.st adaptive time controller advances too much. As a result, we cap the growth factor for faster convergence!
    if dt > 1e3 and fac > 1.0:
        fac = min(fac, 1.5)

    # mild smoothing so dt doesn't jitter
    # Only if fac < 1
    if fac < 1.0:
        fac = 0.8 + 0.2*fac 
    dt = min(dt_max, max(dt_min, dt * fac))

    print(f"[ADAPT] it={it:2d}, eta_rms={eta:.2e}, fac={fac:.2f} -> next dt={dt:.3e}")

    n += 1


# convert to arrays for easier indexing/plotting
ts = np.array(ts, dtype=float)
cw_NR_all = np.array(cw_NR_all, dtype=float)       # (Nt_saved, N_Tetha+1, N_Phi+1)
ceth_NR_all = np.array(ceth_NR_all, dtype=float)   # (Nt_saved, N_Tetha+1, N_Phi+1)
cn2g_ts = np.array(cn2g_ts, dtype=float)
cwg_ts = np.array(cwg_ts, dtype=float)
cethg_ts = np.array(cethg_ts, dtype=float)
cw_surf_ts = np.array(cw_surf_ts, dtype=float)
ceth_surf_ts = np.array(ceth_surf_ts, dtype=float)
cw_rmean_ts = np.array(cw_rmean_ts, dtype=float)       # (Nt_saved, N_R_BASE+1)
ceth_rmean_ts = np.array(ceth_rmean_ts, dtype=float)   # (Nt_saved, N_R_BASE+1)
nW_tot_ts = np.array(nW_tot_ts, dtype=float)
nE_tot_ts = np.array(nE_tot_ts, dtype=float)

# relative drift (%)(From initial total moles and the solver total moles)
# If we have done a good job the dirft percentage should be really low!
W_drift_pct = 100.0 * (nW_tot_ts - nW_tot0) / max(abs(nW_tot0), 1e-30)
E_drift_pct = 100.0 * (nE_tot_ts - nE_tot0) / max(abs(nE_tot0), 1e-30)
#
# PLOTS (time series)
# 
if ts.size:
    # cw and ceth plots at r=N_R mean over tetha and phi
    plt.figure(figsize=(10, 6))
    plt.plot(ts, cw_surf_ts, label="cw @ i=N_R_w (mean over θ,φ)")
    plt.plot(ts, ceth_surf_ts, label="ceth @ i=N_R_eth (mean over θ,φ)")
    plt.xlabel("time [s]")
    plt.ylabel("liquid surface concentration")
    plt.grid(True)
    plt.legend()

    # Gas species plots with time
    plt.figure(figsize=(10, 6))
    plt.plot(ts, cwg_ts[1:], label="cwg")
    plt.plot(ts, cethg_ts[1:], label="cethg")
    plt.plot(ts, cn2g_ts[1:], label="cn2g")
    plt.xlabel("time [s]")
    plt.ylabel("gas concentration [mol/m³]")
    plt.grid(True)
    plt.legend()

    # Pressure in the cylinder vessel with time
    plt.figure(figsize=(10, 6))
    plt.plot(ts, P_ts[1:], label="P_use")
    plt.xlabel("time [s]")
    plt.ylabel("P_use [bar]")
    plt.grid(True)
    plt.legend()

    # Space time heat map for the ethanol concentration
    r_base = r_points  # length N_R_BASE+1

    plt.figure(figsize=(10, 6))
    plt.title("Ethanol in liquid: radial mean vs (r,t)")
    # imshow expects [rows, cols] => [time, radius]
    plt.imshow(
        ceth_rmean_ts,
        aspect="auto",
        origin="lower",
        extent=[r_base[0], r_base[-1], ts[0], ts[-1]]
    )
    plt.xlabel("radius r [m]")
    plt.ylabel("time [s]")
    plt.colorbar(label="⟨c_eth⟩θ,φ  [mol/m³] (or your units)")
    plt.tight_layout()

    # Space time heat map for the water concentration
    plt.figure(figsize=(10, 6))
    plt.title("Water in liquid: radial mean vs (r,t)")
    plt.imshow(
        cw_rmean_ts,
        aspect="auto",
        origin="lower",
        extent=[r_base[0], r_base[-1], ts[0], ts[-1]]
    )
    plt.xlabel("radius r [m]")
    plt.ylabel("time [s]")
    plt.colorbar(label="mean c_w over theta,phi [mol/m^3]")
    plt.tight_layout()

    # Plotting ethanol radial disturbution at selected times
    Nt = ceth_rmean_ts.shape[0]
    idx_t1e3 = int(np.argmin(np.abs(ts - 1e3)))
    pick = sorted(set([0, idx_t1e3, Nt-1]))

    plt.figure(figsize=(10, 6))
    for idx in pick:
        plt.plot(r_base, ceth_rmean_ts[idx, :], label=f"t={ts[idx]:.2e} s")
    plt.xlabel("radius r [m]")
    plt.ylabel("⟨c_eth(r)⟩θ,φ")
    plt.grid(True)
    plt.legend()
    plt.title("Ethanol radial distribution at selected times")

    # Plotting water radial distribution at selected times
    plt.figure(figsize=(10, 6))
    for idx in pick:
        plt.plot(r_base, cw_rmean_ts[idx, :], label=f"t={ts[idx]:.2e} s")
    plt.xlabel("radius r [m]")
    plt.ylabel("mean c_w(r) over theta,phi")
    plt.grid(True)
    plt.legend()
    plt.title("Water radial distribution at selected times")


    # Checking the mass conservation drifts
    plt.figure(figsize=(10, 6))
    plt.plot(ts, W_drift_pct, label="Water total moles drift [%]")
    plt.plot(ts, E_drift_pct, label="Ethanol total moles drift [%]")
    plt.xlabel("time [s]")
    plt.ylabel("relative drift in (n_liq + n_gas) [%]")
    plt.grid(True)
    plt.legend()
    plt.title("Closed-system mass conservation check (before drying logic alters inventory)")
    plt.show()

    # Angular heat maps on the liquid surface
    # theta-time map: average over phi
    # phi-time map: average over theta
    if cw_NR_all.size:
        Nt_w = min(ts.size, cw_NR_all.shape[0])
        ts_w = ts[:Nt_w]
        cw_theta_t = np.mean(cw_NR_all[:Nt_w, :, :], axis=2)  # (time, theta)
        cw_phi_t = np.mean(cw_NR_all[:Nt_w, :, :], axis=1)    # (time, phi)

        plt.figure(figsize=(10, 6))
        plt.title("Water surface: mean over phi vs (theta,t)")
        plt.imshow(
            cw_theta_t,
            aspect="auto",
            origin="lower",
            extent=[tetha_points[0], tetha_points[-1], ts_w[0], ts_w[-1]]
        )
        plt.xlabel("theta [rad]")
        plt.ylabel("time [s]")
        plt.colorbar(label="mean c_w over phi")
        plt.tight_layout()

        plt.figure(figsize=(10, 6))
        plt.title("Water surface: mean over theta vs (phi,t)")
        plt.imshow(
            cw_phi_t,
            aspect="auto",
            origin="lower",
            extent=[phi_points[0], phi_points[-1], ts_w[0], ts_w[-1]]
        )
        plt.xlabel("phi [rad]")
        plt.ylabel("time [s]")
        plt.colorbar(label="mean c_w over theta")
        plt.tight_layout()

    if ceth_NR_all.size:
        Nt_e = min(ts.size, ceth_NR_all.shape[0])
        ts_e = ts[:Nt_e]
        ceth_theta_t = np.mean(ceth_NR_all[:Nt_e, :, :], axis=2)  # (time, theta)
        ceth_phi_t = np.mean(ceth_NR_all[:Nt_e, :, :], axis=1)    # (time, phi)

        plt.figure(figsize=(10, 6))
        plt.title("Ethanol surface: mean over phi vs (theta,t)")
        plt.imshow(
            ceth_theta_t,
            aspect="auto",
            origin="lower",
            extent=[tetha_points[0], tetha_points[-1], ts_e[0], ts_e[-1]]
        )
        plt.xlabel("theta [rad]")
        plt.ylabel("time [s]")
        plt.colorbar(label="mean c_eth over phi")
        plt.tight_layout()

        plt.figure(figsize=(10, 6))
        plt.title("Ethanol surface: mean over theta vs (phi,t)")
        plt.imshow(
            ceth_phi_t,
            aspect="auto",
            origin="lower",
            extent=[phi_points[0], phi_points[-1], ts_e[0], ts_e[-1]]
        )
        plt.xlabel("phi [rad]")
        plt.ylabel("time [s]")
        plt.colorbar(label="mean c_eth over theta")
        plt.tight_layout()

#
#visualize cw/ceth fields on (theta,phi) at a few times

# pick a few indices to plot (e.g. start, middle, end)
if ts.size:
    show_idx = [0, int(0.5*(ts.size-1)), ts.size-1]
    for idx in show_idx:
        plt.figure(figsize=(8, 4))
        plt.title(f"cw at i=N_R_w, t={ts[idx]:.3f} s")
        plt.imshow(cw_NR_all[idx, :, :], aspect="auto", origin="lower")
        plt.xlabel("phi index k")
        plt.ylabel("theta index j")
        plt.colorbar()
        plt.tight_layout()

        plt.figure(figsize=(8, 4))
        plt.title(f"ceth at i=N_R_eth, t={ts[idx]:.3f} s")
        plt.imshow(ceth_NR_all[idx, :, :], aspect="auto", origin="lower")
        plt.xlabel("phi index k")
        plt.ylabel("theta index j")
        plt.colorbar()
        plt.tight_layout()

    # Additional angular diagnostics at final time (surface i = N_R)
    idx = 0
    if cw_NR_all.size:
        cw_surf = cw_NR_all[idx]
        ceth_surf = ceth_NR_all[idx]

        # theta-cut at phi=0
        k0 = 0
        plt.figure(figsize=(8, 4))
        plt.plot(tetha_points, cw_surf[:, k0], label="cw @ phi=0")
        plt.plot(tetha_points, ceth_surf[:, k0], label="ceth @ phi=0")
        plt.xlabel("theta [rad]")
        plt.ylabel("surface concentration")
        plt.title(f"Surface theta-cut at phi=0, t={ts[idx]:.3f} s")
        plt.grid(True)
        plt.legend()

        # phi-cut at mid-theta
        jmid = N_Tetha // 2
        plt.figure(figsize=(8, 4))
        plt.plot(phi_points, cw_surf[jmid, :], label="cw @ theta=mid")
        plt.plot(phi_points, ceth_surf[jmid, :], label="ceth @ theta=mid")
        plt.xlabel("phi [rad]")
        plt.ylabel("surface concentration")
        plt.title(f"Surface phi-cut at theta index {jmid}, t={ts[idx]:.3f} s")
        plt.grid(True)
        plt.legend()

    plt.show()

# Animations

# # Animate ethanol radial profile inside the sphere
# ts = np.asarray(ts)
# r_base = np.asarray(r_base)
# C = np.asarray(ceth_rmean_ts)

# # skip frames if too many (faster rendering)
# stride = max(1, C.shape[0] // 300)   # aim ~300 frames max
# ts_s = ts[::stride]
# C_s  = C[::stride, :]

# # Y-limits ignoring NaNs
# cmin = np.nanmin(C_s)
# cmax = np.nanmax(C_s)

# fig, ax = plt.subplots(figsize=(9, 5))
# line, = ax.plot(r_base, C_s[0], lw=2)

# ax.set_xlabel("radius r [m]")
# ax.set_ylabel("⟨c_eth(r,t)⟩θ,φ  [mol/m³]")
# ax.set_title(f"Ethanol radial profile (t = {ts_s[0]:.3e} s)")
# ax.grid(True)

# # set stable y-limits so plot doesn't jump
# ax.set_ylim(cmin - 0.05*(cmax-cmin), cmax + 0.05*(cmax-cmin))
# ax.set_xlim(r_base[0], r_base[-1])

# def update(frame):
#     y = C_s[frame]
#     line.set_ydata(y)
#     ax.set_title(f"Ethanol radial profile (t = {ts_s[frame]:.3e} s)")
#     return (line,)

# ani = FuncAnimation(fig, update, frames=len(ts_s), interval=40, blit=True)

# plt.show()

# # ---- Saving (optional) ----
# # MP4 (needs ffmpeg installed)
# # ani.save("ethanol_radial.mp4", fps=25, dpi=160)

# # GIF (needs pillow installed)
# # ani.save("ethanol_radial.gif", fps=20, dpi=120, writer="pillow")

# # Animate ethanol on the sponge surface (θ–φ map)
# # ceth_NR_all: shape (Nt, N_Tetha+1, N_Phi+1)
# # ts: shape (Nt,)

# ts = np.asarray(ts)
# Surf = np.asarray(ceth_NR_all)

# stride = max(1, Surf.shape[0] // 300)
# ts_s = ts[::stride]
# Surf_s = Surf[::stride, :, :]

# vmin = np.nanmin(Surf_s)
# vmax = np.nanmax(Surf_s)

# fig, ax = plt.subplots(figsize=(7.5, 4.5))
# im = ax.imshow(Surf_s[0], origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
# cb = plt.colorbar(im, ax=ax)
# cb.set_label("c_eth at r = R_sponge [mol/m³]")

# ax.set_xlabel("phi index k")
# ax.set_ylabel("theta index j")
# ax.set_title(f"Surface ethanol field (t = {ts_s[0]:.3e} s)")

# def update(frame):
#     im.set_data(Surf_s[frame])
#     ax.set_title(f"Surface ethanol field (t = {ts_s[frame]:.3e} s)")
#     return (im,)

# ani = FuncAnimation(fig, update, frames=len(ts_s), interval=40, blit=True)
# plt.show()

# # Save if you want:
# # ani.save("ethanol_surface.mp4", fps=25, dpi=160)
# # ani.save("ethanol_surface.gif", fps=20, dpi=120, writer="pillow")
