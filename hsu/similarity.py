"""Pairwise band-similarity measures: entropy, NMI, VI and SSIM.

H(X)        : Shannon entropy of a band (histogram, `bins` levels)
MI(X,Y)     = H(X) + H(Y) - H(X,Y)
NMI(X,Y)    = 2 MI / (H(X) + H(Y))              in [0, 1]   (1 = identical information)
VI(X,Y)     = H(X,Y) - MI(X,Y)                   (a true metric)
VI_n(X,Y)   = VI / H(X,Y)                        in [0, 1]   (1 = completely different)
SSIM(X,Y)   : structural similarity of the two band images (Wang et al. 2004)
"""
import os
import numpy as np
from skimage.metrics import structural_similarity

CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")


def _quantise(Y, bins):
    lo = np.percentile(Y, 0.5, axis=1, keepdims=True)
    hi = np.percentile(Y, 99.5, axis=1, keepdims=True)
    Z = (Y - lo) / (hi - lo + 1e-12)
    return np.clip((Z * bins).astype(int), 0, bins - 1)


def info_matrices(Y, bins=64, max_pixels=20000, seed=0):
    B, N = Y.shape
    if N > max_pixels:
        cols = np.random.default_rng(seed).choice(N, max_pixels, replace=False)
        Y = Y[:, cols]
        N = max_pixels
    Q = _quantise(Y, bins)
    H = np.zeros(B)
    for i in range(B):
        p = np.bincount(Q[i], minlength=bins) / N
        p = p[p > 0]
        H[i] = -np.sum(p * np.log2(p))
    Hj = np.zeros((B, B))
    for i in range(B):
        codes = Q[i][None, :] * bins + Q[i:]                # (B-i, N)
        for jj, c in enumerate(codes):
            p = np.bincount(c, minlength=bins * bins) / N
            p = p[p > 0]
            Hj[i, i + jj] = Hj[i + jj, i] = -np.sum(p * np.log2(p))
    MI = H[:, None] + H[None, :] - Hj
    NMI = 2 * MI / (H[:, None] + H[None, :] + 1e-12)
    VI = Hj - MI
    VIn = VI / (Hj + 1e-12)
    np.fill_diagonal(NMI, 1.0)
    np.fill_diagonal(VIn, 0.0)
    return H, NMI, VI, VIn


def ssim_matrix(Y, rows, cols, max_side=160):
    from .data import to_image
    B = Y.shape[0]
    img = to_image(Y, rows, cols)
    step = int(np.ceil(max(rows, cols) / max_side))
    img = img[:, ::step, ::step]
    lo = img.min(axis=(1, 2), keepdims=True)
    hi = img.max(axis=(1, 2), keepdims=True)
    img = (img - lo) / (hi - lo + 1e-12)
    S = np.eye(B)
    for i in range(B):
        for j in range(i + 1, B):
            S[i, j] = S[j, i] = structural_similarity(img[i], img[j], data_range=1.0)
    return S


def get_all(d):
    """Compute (or load cached) similarity matrices for dataset dict d."""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, d["name"].replace(" ", "_") + "_sim.npz")
    if os.path.exists(f):
        z = np.load(f)
        return {k: z[k] for k in z.files}
    H, NMI, VI, VIn = info_matrices(d["Y"])
    SSIM = ssim_matrix(d["Y"], d["rows"], d["cols"])
    out = dict(H=H, NMI=NMI, VI=VI, VIn=VIn, SSIM=SSIM, var=d["Y"].var(axis=1))
    np.savez(f, **out)
    return out
