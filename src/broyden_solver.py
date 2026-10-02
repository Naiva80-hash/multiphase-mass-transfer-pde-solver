# Broyden3_fastLU.py  (Paste & run)
# SAME solver as yours, ONLY change: use fast in-place LU instead of (p,L,U) LU.

import numpy as np
from jacobian_coloring import num_jacobe_colored
from numerical_jacobian import  num_jacobe_fast
from norms import L2norm


def clamp_to_zero(c, threshold=1e-6):
    """
    This function ensures the concentration 'c' is never below the threshold.
    """
    return max(c, threshold)

def lu_factor_fast(A, pivot_tol=1e-12):
    A = np.array(A, dtype=float, copy=True, order="C")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"lu_factor_fast expects square matrix, got {A.shape}")
    n = A.shape[0]

    scale = np.linalg.norm(A, ord=np.inf)
    tiny = pivot_tol * max(1.0, scale)

    p = np.arange(n)

    for k in range(n - 1):
        imax = k + np.argmax(np.abs(A[k:, k]))
        if imax != k:
            A[[k, imax], :] = A[[imax, k], :]
            p[[k, imax]] = p[[imax, k]]

        piv = A[k, k]
        if abs(piv) < tiny:
            raise ValueError(f"Singular/ill-conditioned pivot at col {k}: |piv|={abs(piv):.3e}, tiny={tiny:.3e}")

        A[k+1:, k] /= piv
        A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])

    if abs(A[-1, -1]) < tiny:
        raise ValueError(f"Singular/ill-conditioned last pivot: |piv|={abs(A[-1,-1]):.3e}, tiny={tiny:.3e}")

    return A, p


def lu_solve_fast(LU, p, b):
    LU = np.asarray(LU, dtype=float)
    b = np.asarray(b, dtype=float)
    vec = (b.ndim == 1)
    if vec:
        b = b.reshape(-1, 1)

    n = LU.shape[0]
    if b.shape[0] != n:
        raise ValueError(f"lu_solve_fast: b has shape {b.shape}, expected ({n}, k)")

    # Apply permutation: Pb
    x = b[p, :].copy()

    # Forward solve L y = Pb (L has 1s on diagonal)
    for i in range(n):
        x[i, :] -= LU[i, :i] @ x[:i, :]

    # Back solve U x = y
    for i in range(n - 1, -1, -1):
        x[i, :] -= LU[i, i+1:] @ x[i+1:, :]
        x[i, :] /= LU[i, i]

    return x.ravel() if vec else x


def lu_solve_transpose_fast(LU, p, b):
    """
    Solve (A^T) x = b given PA=LU.
    A = P^{-1} L U  => A^T = U^T L^T P^{-T}
    Steps:
      1) w = P^{-T} b
      2) solve U^T z = w
      3) solve L^T x = z
    """
    LU = np.asarray(LU, dtype=float)
    b = np.asarray(b, dtype=float)
    vec = (b.ndim == 1)
    if vec:
        b = b.reshape(-1, 1)

    n = LU.shape[0]
    if b.shape[0] != n:
        raise ValueError(f"lu_solve_transpose_fast: b has shape {b.shape}, expected ({n}, k)")

    invp = np.empty_like(p)
    invp[p] = np.arange(n)

    # w = P^{-T} b
    w = b[invp, :]

    # Solve U^T z = w (U^T is lower)
    z = np.zeros_like(w)
    for i in range(n):
        # U^T(i,0:i) = U(0:i,i) which is LU[0:i, i]
        z[i, :] = (w[i, :] - (LU[:i, i].reshape(1, -1) @ z[:i, :])) / (LU[i, i] + 1e-300)

    # Solve L^T x = z (L^T is upper, diag=1)
    x = np.zeros_like(z)
    for i in range(n - 1, -1, -1):
        # L^T(i,i+1:) = L(i+1:,i) which is LU[i+1:, i]
        x[i, :] = z[i, :] - (LU[i+1:, i].reshape(1, -1) @ x[i+1:, :])

    return x.ravel() if vec else x




def build_groups_greedy(rows_by_col):
    """
    Checking whether two columns can be perturbed simultaneously or not
    """
    n = len(rows_by_col)

    # reverse look-up table
    # rows_by_col[c] tells us for column c, which residual rows
    # depend on it
    # rows_to_col[r] for residual row r, which columns influence it.
    row_to_cols = {}
    for c in range(n):
        # Loop through all residual tha tcolumn c affects
        for r in rows_by_col[c]:
            # if r not in dictionary yet, create it with an empty list
            # append the current column c to that list
            row_to_cols.setdefault(int(r), []).append(c)

    # In order that we color the most constrainted columns first
    # range(n) Create the list of all column indicies
    # rows_by_col[c] rows wher column c is nonzero
    # len(rows_by_col[c]) how many rows it affects
    # sort it from largest to smallest
    order = sorted(range(n), key=lambda c: len(rows_by_col[c]), reverse=True)

    #List of -1
    color = [-1]*n
    # Loop through hardest to easiest columns to color
    for c in order:
        # Set of used colors
        used = set()
        # Check all the rows that this column affects
        for r in rows_by_col[c]:
            # Finding all columns that this specific row affects 
            for nb in row_to_cols[int(r)]:
                # Finding the element of this color in colors list
                cc = color[nb]
                # If it is assigned to a color add it to used set
                if cc != -1:
                    used.add(cc)
        # smallest available color
        col = 0
        # If col is used add 1 till it is not in the used anymore
        while col in used:
            col += 1
        # Else assign that column with a color(smallest number possible)
        color[c] = col

    # Generating the numer of colors and groups list of lists
    ncolors = max(color) + 1
    groups = [[] for _ in range(ncolors)]
    # Loop in colors list and add the specific colors for each specific group 
    for c, col in enumerate(color):
        groups[col].append(c)


    return [np.array(g, dtype=int) for g in groups if len(g) > 0], ncolors







#
# Limited-memory inverse Broyden operator (SAME math, fast LU backend)
# 
class LBroydenH:
    def __init__(self, LU, p, m=12):
        self.LU = LU
        self.p  = p
        self.m  = int(m)

        self.U_list = []
        self.V_list = []
        self.D_list = []

    def H0_apply(self, g):
        return -lu_solve_fast(self.LU, self.p, g)

    def H0T_apply(self, g):
        return -lu_solve_transpose_fast(self.LU, self.p, g)

    def apply(self, g):
        g = np.asarray(g, dtype=float)
        y = self.H0_apply(g)
        for u, v, den in zip(self.U_list, self.V_list, self.D_list):
            y = y - u * ((v @ g) / (den + 1e-300))
        return y

    def apply_T(self, g):
        g = np.asarray(g, dtype=float)
        y = self.H0T_apply(g)
        for u, v, den in zip(self.U_list, self.V_list, self.D_list):
            y = y - v * ((u @ g) / (den + 1e-300))
        return y

    def update(self, u, v, den):
        self.U_list.append(np.asarray(u, dtype=float).copy())
        self.V_list.append(np.asarray(v, dtype=float).copy())
        self.D_list.append(float(den))

        if len(self.U_list) > self.m:
            self.U_list.pop(0)
            self.V_list.pop(0)
            self.D_list.pop(0)

def liquid_nonnegative(x_trial, N_tot):
    return np.all(x_trial[:2*N_tot] >= 0.0)
#
# Solver (UNCHANGED except LU line)
# 
def householder_like_lu(
    f, x0,
    eps=1e-8, max_outer_num=50, max_inner_num=20,
    rel_step=1e-7,
    m_hist=12,
    verbose=False,
    N_tot = None
):
    x = np.asarray(x0, dtype=float)
    Fk = np.asarray(f(x), dtype=float)

    recomp_J = True
    outer_iter = 0
    Fk_norm = float(L2norm(Fk))

    H = None

    rebuild_count = 0
    max_rebuilds = 2

    while (Fk_norm > eps) and (outer_iter < max_outer_num):

        if verbose:
            print(f"Iter {outer_iter:3d}  ||F|| = {Fk_norm:.3e}")

        if recomp_J or outer_iter == 0 or H is None:
            rebuild_count += 1
            if rebuild_count > max_rebuilds:
                if verbose:
                    print("  [STOP] Jacobian recomputed too many times -> no convergence")
                return x, Fk, False, outer_iter

            J = num_jacobe_colored(f, x, Fk, rows_by_col, groups, rel_step=rel_step)
            try:
                LU, p = lu_factor_fast(J, pivot_tol=1e-12)
            except ValueError as e:
                if verbose:
                    print(f"  [FAIL] LU factorization: {e}")
                return x, Fk, False, outer_iter
            H = LBroydenH(LU, p, m=m_hist)
            recomp_J = False

            if verbose:
                print(f"  [rebuild] J colored + FAST LU (count={rebuild_count})")

        F_k2 = float(Fk @ Fk)

        delta_x = H.apply(Fk)
        s = 1.0

        x_k1 = x + s * delta_x
        F_k1 = np.asarray(f(x_k1), dtype=float)
        F_k12 = float(F_k1 @ F_k1)

        inner_iter = 0
        #(((not liquid_nonnegative(x_k1, N_tot)) or
        while  ((F_k12 > F_k2)and (inner_iter < max_inner_num)):

            # Use your Householder-like step reduction (no halving)
            eta = F_k12 / (F_k2 + 1e-300)
            s_new = ((1.0 + 6.0 * eta) ** 0.5 - 1.0) / (3.0 * eta + 1e-300)
            s = min(s_new, 0.8*s)
            if s < 1e-6:
                break

            x_k1 = x + s * delta_x
            F_k1 = np.asarray(f(x_k1), dtype=float)
            F_k12 = float(F_k1 @ F_k1)

            inner_iter += 1

        # If still bad (negative or not decreasing) => rebuild J
        #not liquid_nonnegative(x_k1, N_tot)) or
        if  (F_k12 >= F_k2):
            recomp_J = True
            if verbose:
                print("  Line search failed (negativity or no decrease); recomputing J next iteration.")
            outer_iter += 1
            continue

        Y = F_k1 - Fk
        HY = H.apply(Y)

        u = HY + s * delta_x
        v = H.apply_T(delta_x)
        den = float(delta_x @ HY)

        if abs(den) < 1e-20:
            x, Fk = x_k1, F_k1
            Fk_norm = float(L2norm(Fk))
            recomp_J = True
            if verbose:
                print("  tiny denom; forcing J rebuild next iter.")
            outer_iter += 1
            continue

        H.update(u, v, den)

        x, Fk = x_k1, F_k1
        Fk_norm = float(L2norm(Fk))
        outer_iter += 1

        rebuild_count = 0

        if verbose:
            print(f"  accepted: s={s:.3e}, inner={inner_iter}, denom={den:.3e}")

    return x, Fk, (float(L2norm(Fk)) <= eps), outer_iter
