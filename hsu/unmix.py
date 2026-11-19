"""Linear unmixing: VCA / N-FINDR endmember extraction, FCLS abundances, and evaluation metrics.

Linear mixing model:  Y = M A + noise,  A >= 0,  sum(A, axis=0) = 1
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------- endmember extraction
def vca(Y, p, seed=0, snr_input=None):
    """Vertex Component Analysis (Nascimento & Bioucas-Dias, 2005).

    Y : (B, N) data, p : number of endmembers.
    Returns (E, idx) : E (B, p) endmembers, idx (p,) indices of the chosen pixels.
    """
    rng = np.random.default_rng(seed)
    B, N = Y.shape
    Ym = Y.mean(axis=1, keepdims=True)
    Yc = Y - Ym
    U, s, _ = np.linalg.svd(Yc @ Yc.T / N)
    Ud = U[:, :p]
    x_p = Ud.T @ Yc
    P_y = np.sum(Y ** 2) / N
    P_x = np.sum(x_p ** 2) / N + float(np.sum(Ym ** 2))
    if snr_input is None:
        snr = 10 * np.log10(max(P_x - p / B * P_y, 1e-12) / max(P_y - P_x, 1e-12))
    else:
        snr = snr_input
    snr_th = 15 + 10 * np.log10(p)

    if snr < snr_th:                       # low SNR: projective projection onto (p-1)-subspace
        d = p - 1
        Ud = U[:, :d]
        x = Ud.T @ Yc
        c = np.sqrt(np.max(np.sum(x ** 2, axis=0)))
        Yp = np.vstack([x, c * np.ones((1, N))])
    else:                                  # high SNR: projection onto p-subspace
        d = p
        U2, _, _ = np.linalg.svd(Y @ Y.T / N)
        Ud = U2[:, :d]
        x = Ud.T @ Y
        u = x.mean(axis=1, keepdims=True)
        denom = np.sum(x * u, axis=0, keepdims=True)
        denom[np.abs(denom) < 1e-12] = 1e-12
        Yp = x / denom

    idx = np.zeros(p, dtype=int)
    A = np.zeros((p, p))
    A[-1, 0] = 1
    for i in range(p):
        w = rng.standard_normal((p, 1))
        f = w - A @ np.linalg.pinv(A) @ w
        f = f / (np.linalg.norm(f) + 1e-12)
        v = f.T @ Yp
        idx[i] = int(np.argmax(np.abs(v)))
        A[:, i] = Yp[:, idx[i]]
    return Y[:, idx], idx


def nfindr(Y, p, seed=0, max_iter=3):
    """N-FINDR (Winter, 1999): maximise simplex volume in a (p-1)-dim PCA space.
    Vectorised: volume is linear in the replaced column, via cofactors."""
    rng = np.random.default_rng(seed)
    B, N = Y.shape
    Ym = Y.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd((Y - Ym) @ (Y - Ym).T / N)
    Z = U[:, :p - 1].T @ (Y - Ym)                      # (p-1, N)
    Zh = np.vstack([np.ones((1, N)), Z])               # (p, N)
    idx = rng.choice(N, p, replace=False)
    for _ in range(max_iter):
        changed = False
        for j in range(p):
            E = Zh[:, idx].copy()
            # det(E with column j replaced by z) = cof_j . z
            cof = np.zeros(p)
            for r in range(p):
                minor = np.delete(np.delete(E, r, axis=0), j, axis=1)
                cof[r] = ((-1) ** (r + j)) * np.linalg.det(minor)
            vol = np.abs(cof @ Zh)
            best = int(np.argmax(vol))
            if vol[best] > np.abs(np.linalg.det(E)) * (1 + 1e-9) and best not in idx:
                idx[j] = best
                changed = True
        if not changed:
            break
    return Y[:, idx], idx


# ---------------------------------------------------------------- abundance estimation
def _proj_simplex(V):
    """Euclidean projection of each column of V onto the probability simplex."""
    p, N = V.shape
    U = -np.sort(-V, axis=0)
    css = np.cumsum(U, axis=0) - 1
    ind = np.arange(1, p + 1)[:, None]
    cond = U - css / ind > 0
    rho = p - 1 - np.argmax(cond[::-1], axis=0)
    theta = css[rho, np.arange(N)] / (rho + 1)
    return np.maximum(V - theta, 0)


def fcls(Y, E, n_iter=150, A0=None):
    """Fully Constrained Least Squares, solved for all pixels at once
    with accelerated projected gradient (FISTA) onto the simplex.

    min_a 0.5 || y - E a ||^2  s.t.  a >= 0, 1^T a = 1
    """
    EtE = E.T @ E
    EtY = E.T @ Y
    L = np.linalg.eigvalsh(EtE).max() + 1e-12
    p, N = E.shape[1], Y.shape[1]
    A = np.full((p, N), 1.0 / p) if A0 is None else A0.copy()
    Z, t = A.copy(), 1.0
    for _ in range(n_iter):
        A_new = _proj_simplex(Z - (EtE @ Z - EtY) / L)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        Z = A_new + ((t - 1) / t_new) * (A_new - A)
        A, t = A_new, t_new
    return A


# ---------------------------------------------------------------- metrics
def sad_matrix(E, M):
    """Spectral angle (radians) between every estimated (cols of E) and true (cols of M) endmember."""
    En = E / (np.linalg.norm(E, axis=0, keepdims=True) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-12)
    return np.arccos(np.clip(Mn.T @ En, -1, 1))          # (p_true, p_est)


def match(E, M):
    """Hungarian matching of estimated to true endmembers by SAD. Returns perm s.t. E[:, perm] ~ M."""
    S = sad_matrix(E, M)
    r, c = linear_sum_assignment(S)
    perm = np.empty(M.shape[1], dtype=int)
    perm[r] = c
    return perm, S[r, c]


def rmse_abund(A_hat, A):
    return float(np.sqrt(np.mean((A_hat - A) ** 2)))


def rmse_recon(Y, E, A):
    return float(np.sqrt(np.mean((Y - E @ A) ** 2)))


def unmix_pipeline(Y, bands, p, M_gt=None, A_gt=None, seed=0, extractor="vca", n_iter=150):
    """Fixed pipeline where only the band subset changes.

    1. endmember extraction (VCA / N-FINDR) on the selected bands  -> pixel indices
    2. full-band endmember spectra are read back at those pixels   (for SAD + reconstruction)
    3. FCLS abundances computed using only the selected bands
    """
    bands = np.sort(np.asarray(bands))
    Ys = Y[bands]
    ext = vca if extractor == "vca" else nfindr
    Es, idx = ext(Ys, p, seed=seed)
    A_hat = fcls(Ys, Es, n_iter=n_iter)
    E_full = Y[:, idx]
    out = dict(idx=idx, E=E_full, A=A_hat, bands=bands)
    out["RE"] = rmse_recon(Y, E_full, A_hat)
    if M_gt is not None:
        perm, sads = match(E_full, M_gt)
        out["perm"] = perm
        out["SAD_each"] = sads
        out["SAD"] = float(np.mean(sads))
        out["E"] = E_full[:, perm]
        out["A"] = A_hat[perm]
        if A_gt is not None:
            out["RMSE"] = rmse_abund(out["A"], A_gt)
            out["RMSE_each"] = np.sqrt(np.mean((out["A"] - A_gt) ** 2, axis=1))
    return out
