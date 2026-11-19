"""Run the full benchmark for one or more datasets.

usage:  python run_experiments.py samson jasper urban cuprite
Outputs (results/):
    sweep_<ds>.csv        method x k x run  -> SAD, RMSE, RE, times
    bands_<ds>.json       selected band indices per method / k
    corr_<ds>.csv         random-subset study: subset NMI/SSIM/VI/H/GSR score vs unmixing error
    ablation_<ds>.csv     VCA vs N-FINDR at k = 15
    ga_hist_<ds>.json     GA convergence curve
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from hsu import data, similarity, unmix
from hsu import band_selection as bs

OUT = "results"
KS = [5, 10, 15, 20, 25, 30]
RUNS = 5          # VCA seeds
GA_SEEDS = 3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run_dataset(name):
    os.makedirs(OUT, exist_ok=True)
    d = data.load(name)
    Y, p, M, A = d["Y"], d["p"], d["M"], d["A"]
    B = Y.shape[0]
    t0 = time.time()
    sim = similarity.get_all(d)
    t_sim = time.time() - t0
    ks = [k for k in KS if k > p] if p > 5 else KS
    rows, bands_json = [], {}

    # full-band reference
    for r in range(RUNS):
        t = time.time()
        o = unmix.unmix_pipeline(Y, np.arange(B), p, M, A, seed=r)
        rows.append(dict(dataset=d["name"], method="Full bands", k=B, run=r, SAD=o["SAD"],
                         RMSE=o.get("RMSE", np.nan), RE=o["RE"], t_bs=0.0, t_unmix=time.time() - t))
    log(name, "full", np.mean([x["SAD"] for x in rows]))

    for m in bs.METHODS:
        bands_json[m] = {}
        for k in ks:
            cache = {}
            for r in range(RUNS):
                sseed = (r % GA_SEEDS) if m == "GSR-GA-U" else r
                key = sseed if m in bs.STOCHASTIC else 0
                if key not in cache:
                    t = time.time()
                    cache[key] = (bs.select(m, d, sim, k, seed=sseed), time.time() - t)
                b, tb = cache[key]
                t = time.time()
                o = unmix.unmix_pipeline(Y, b, p, M, A, seed=r)
                rows.append(dict(dataset=d["name"], method=m, k=k, run=r, SAD=o["SAD"],
                                 RMSE=o.get("RMSE", np.nan), RE=o["RE"], t_bs=tb,
                                 t_unmix=time.time() - t))
            bands_json[m][k] = [int(x) for x in cache[min(cache)][0]]
        df = pd.DataFrame(rows)
        log(name, m, df[df.method == m].groupby("k")[["SAD", "RMSE", "RE"]].mean().round(3).to_dict("list"))
    pd.DataFrame(rows).to_csv(f"{OUT}/sweep_{name}.csv", index=False)
    json.dump(dict(bands=bands_json, B=B, t_sim=t_sim), open(f"{OUT}/bands_{name}.json", "w"))

    # ---- GA convergence history (k=15)
    _, hist = bs.gsr_ga_u(B, 15, sim=sim, Y=Y, p=p, seed=0, return_history=True)
    json.dump([float(h) for h in hist], open(f"{OUT}/ga_hist_{name}.json", "w"))

    # ---- metric-vs-accuracy correlation study on random subsets
    rng = np.random.default_rng(7)
    W = bs.gsr_weights(sim)
    crow = []
    k = 15
    for t in range(80):
        b = np.sort(rng.choice(B, k, replace=False))
        iu = np.triu_indices(k, 1)
        sub = np.ix_(b, b)
        feats = dict(NMI=sim["NMI"][sub][iu].mean(), SSIM=sim["SSIM"][sub][iu].mean(),
                     VI=sim["VIn"][sub][iu].mean(), Entropy=sim["H"][b].mean(),
                     GSR=W[b].sum(axis=1).mean(), Spread=np.std(b) / B)
        os_ = [unmix.unmix_pipeline(Y, b, p, M, A, seed=s) for s in range(2)]
        crow.append(dict(trial=t, **feats, SAD=np.mean([o["SAD"] for o in os_]),
                         RMSE=np.mean([o.get("RMSE", np.nan) for o in os_]),
                         RE=np.mean([o["RE"] for o in os_])))
    pd.DataFrame(crow).to_csv(f"{OUT}/corr_{name}.csv", index=False)
    log(name, "corr done")

    # ---- ablation: VCA vs N-FINDR at k = 15
    arow = []
    for m in bs.METHODS:
        b = np.array(bands_json[m][15])
        for ext in ("vca", "nfindr"):
            os_ = [unmix.unmix_pipeline(Y, b, p, M, A, seed=s, extractor=ext) for s in range(2)]
            arow.append(dict(method=m, extractor=ext.upper().replace("NFINDR", "N-FINDR"),
                             SAD=np.mean([o["SAD"] for o in os_]),
                             RMSE=np.mean([o.get("RMSE", np.nan) for o in os_]),
                             RE=np.mean([o["RE"] for o in os_])))
    pd.DataFrame(arow).to_csv(f"{OUT}/ablation_{name}.csv", index=False)
    log(name, "DONE")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run_dataset(n)
