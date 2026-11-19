"""Band selection (BS) algorithms compared in this study.

Baselines
    Uniform   : equally spaced bands
    Random    : random subset (averaged over seeds)
    MaxVar    : top-k band variance
    IE        : information entropy, top-k bands with largest entropy
    OPBS      : orthogonal projection band selection (Zhang et al., TGRS 2018)
    WaLuMI    : Ward-linkage-style clustering of adjacent bands using MI distance,
                one representative band per cluster (Martinez-Uso et al., TGRS 2007)

Similarity-ranking family (department's prior work)
    SR-NMI-VI : similarity ranking with NMI (representativeness) and VI (distinctness),
                our re-implementation of Nadipelli et al., MIGARS 2024
    GSR-Hub   : graph-relational "weighted reach" ranking of MO-GSR-BS, Phase 1
                w_ij = (a*NMI + (1-a)*SSIM) * VI_n,  R_i = sum_j w_ij
    GSR-GA-U  : MO-GSR-BS Phase 2 adapted to unmixing (proposed): evolutionary search seeded
                80 % from GSR hubs, fitness = unsupervised unmixing reconstruction error
                (no ground truth is used)
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from . import unmix


def uniform(B, k, **_):
    return np.unique(np.round(np.linspace(0, B - 1, k)).astype(int))


def random_bs(B, k, seed=0, **_):
    return np.sort(np.random.default_rng(1000 + seed).choice(B, k, replace=False))


def maxvar(B, k, sim, **_):
    return np.sort(np.argsort(-sim["var"])[:k])


def entropy_bs(B, k, sim, **_):
    return np.sort(np.argsort(-sim["H"])[:k])


def opbs(B, k, Y, **_):
    """Greedy: pick band with largest norm, then repeatedly the band with the largest
    residual after projecting onto the span of the already selected bands."""
    X = Y - Y.mean(axis=1, keepdims=True)
    R = X.copy()
    sel = [int(np.argmax(np.sum(R ** 2, axis=1)))]
    for _ in range(k - 1):
        q = R[sel[-1]] / (np.linalg.norm(R[sel[-1]]) + 1e-12)
        R = R - np.outer(R @ q, q)
        n = np.sum(R ** 2, axis=1)
        n[sel] = -1
        sel.append(int(np.argmax(n)))
    return np.sort(sel)


def walumi(B, k, sim, **_):
    """Hierarchical clustering on 1-NMI distance, contiguous-band friendly; pick per cluster
    the band with the highest mean NMI to the cluster members."""
    D = 1 - sim["NMI"]
    iu = np.triu_indices(B, 1)
    Z = linkage(D[iu], method="average")
    lab = fcluster(Z, k, criterion="maxclust")
    sel = []
    for c in np.unique(lab):
        m = np.where(lab == c)[0]
        sel.append(m[np.argmax(sim["NMI"][np.ix_(m, m)].mean(axis=1))])
    sel = np.array(sel)
    if len(sel) < k:                                  # top up with high-entropy bands
        extra = [b for b in np.argsort(-sim["H"]) if b not in sel][: k - len(sel)]
        sel = np.concatenate([sel, extra])
    return np.sort(sel[:k]).astype(int)


def sr_nmi_vi(B, k, sim, **_):
    """Representativeness r_i = mean_j NMI(i,j). Greedy selection maximising
    r_i * min_{s in S} VI_n(i, s): bands that represent the cube well AND differ from the
    bands already chosen."""
    r = sim["NMI"].mean(axis=1)
    sel = [int(np.argmax(r * sim["H"] / sim["H"].max()))]
    for _ in range(k - 1):
        score = r * sim["VIn"][:, sel].min(axis=1)
        score[sel] = -1
        sel.append(int(np.argmax(score)))
    return np.sort(sel)


def gsr_weights(sim, alpha=0.5):
    S = alpha * sim["NMI"] + (1 - alpha) * sim["SSIM"]
    W = S * sim["VIn"]
    np.fill_diagonal(W, 0)
    return W


def gsr_hub(B, k, sim, alpha=0.5, **_):
    R = gsr_weights(sim, alpha).sum(axis=1)
    return np.sort(np.argsort(-R)[:k])


def _fitness(Ysub, bands, p, seed=0):
    E, idx = unmix.vca(Ysub[bands], p, seed=seed)
    A = unmix.fcls(Ysub[bands], E, n_iter=60)
    return unmix.rmse_recon(Ysub, Ysub[:, idx], A)


def gsr_ga_u(B, k, sim, Y, p, seed=0, pop=16, gens=12, n_pix=2500, pc=0.7, pm=0.2, alpha=0.5,
             return_history=False, **_):
    """Unmixing-aware evolutionary refinement of GSR hubs (fixed subset size k).
    Population: 80 % sampled from the top-2k hubs, 20 % random. Single-point crossover,
    random-replacement mutation, elitism. Fitness averaged over 2 VCA seeds."""
    rng = np.random.default_rng(seed)
    cols = rng.choice(Y.shape[1], min(n_pix, Y.shape[1]), replace=False)
    Ysub = Y[:, cols]
    R = gsr_weights(sim, alpha).sum(axis=1)
    hubs = np.argsort(-R)[: min(B, max(2 * k, k + 5))]

    def fix(ind):
        ind = list(dict.fromkeys(int(b) for b in ind))
        while len(ind) < k:
            b = int(rng.integers(B))
            if b not in ind:
                ind.append(b)
        return np.sort(np.array(ind[:k]))

    P = [np.sort(np.argsort(-R)[:k])]
    while len(P) < pop:
        if len(P) < int(0.8 * pop):
            P.append(np.sort(rng.choice(hubs, k, replace=False)))
        else:
            P.append(np.sort(rng.choice(B, k, replace=False)))
    cache = {}

    def fit(ind):
        key = tuple(ind)
        if key not in cache:
            cache[key] = np.mean([_fitness(Ysub, ind, p, seed=s) for s in (0, 1)])
        return cache[key]

    hist = []
    for g in range(gens):
        F = np.array([fit(ind) for ind in P])
        order = np.argsort(F)
        hist.append(F[order[0]])
        elite = [P[i] for i in order[:2]]
        children = list(elite)
        while len(children) < pop:
            a, b = [P[min(rng.choice(pop, 2), key=lambda i: F[i])] for _ in range(2)]   # tournament
            if rng.random() < pc:
                cut = int(rng.integers(1, k))
                child = np.concatenate([a[:cut], b[cut:]])
            else:
                child = a.copy()
            child = list(child)
            for j in range(k):
                if rng.random() < pm / k * 3:
                    child[j] = int(rng.choice(hubs)) if rng.random() < 0.5 else int(rng.integers(B))
            children.append(fix(child))
        P = children
    F = np.array([fit(ind) for ind in P])
    best = P[int(np.argmin(F))]
    if return_history:
        return best, hist
    return best


METHODS = {
    "Uniform": uniform,
    "Random": random_bs,
    "MaxVar": maxvar,
    "IE": entropy_bs,
    "OPBS": opbs,
    "WaLuMI": walumi,
    "SR-NMI-VI": sr_nmi_vi,
    "GSR-Hub": gsr_hub,
    "GSR-GA-U": gsr_ga_u,
}
STOCHASTIC = {"Random", "GSR-GA-U"}


def select(method, d, sim, k, seed=0):
    B = d["Y"].shape[0]
    return METHODS[method](B, k, sim=sim, Y=d["Y"], p=d["p"], seed=seed)
