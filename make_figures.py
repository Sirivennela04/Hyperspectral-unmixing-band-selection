"""Generate all tables and figures from results/ into figures/ and results/tables.md"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hsu import data, similarity, unmix

R, F = "results", "figures"
os.makedirs(F, exist_ok=True)
DS = [("samson", "Samson"), ("jasper", "Jasper Ridge"), ("urban", "Urban"), ("cuprite", "Cuprite")]
DS = [x for x in DS if os.path.exists(f"{R}/sweep_{x[0]}.csv")]
ORDER = ["Uniform", "Random", "MaxVar", "IE", "OPBS", "WaLuMI", "SR-NMI-VI", "GSR-Hub", "GSR-GA-U"]
COL = dict(zip(ORDER, ["#7f7f7f", "#bcbd22", "#8c564b", "#e377c2", "#17becf", "#9467bd",
                       "#2ca02c", "#ff7f0e", "#d62728"]))
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 160,
                     "savefig.bbox": "tight"})

sweeps = {n: pd.read_csv(f"{R}/sweep_{n}.csv") for n, _ in DS}
md = []

# ------------------------------------------------------------------ metric vs k curves
for metric, label in [("SAD", "Mean SAD (rad)  ↓"), ("RMSE", "Abundance RMSE  ↓"), ("RE", "Reconstruction error  ↓")]:
    dss = [(n, t) for n, t in DS if sweeps[n][metric].notna().any()]
    fig, axs = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3.6))
    axs = np.atleast_1d(axs)
    for ax, (n, t) in zip(axs, dss):
        df = sweeps[n]
        full = df[df.method == "Full bands"][metric].mean()
        for m in ORDER:
            g = df[df.method == m].groupby("k")[metric].mean()
            ax.plot(g.index, g.values, "-o", ms=3, lw=2.2 if m.startswith("GSR") or m == "SR-NMI-VI" else 1.2,
                    color=COL[m], label=m)
        ax.axhline(full, ls="--", c="k", lw=1.2, label="Full bands")
        ax.set_title(t)
        ax.set_xlabel("Number of selected bands k")
    axs[0].set_ylabel(label)
    axs[-1].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(f"{F}/{metric.lower()}_vs_k.png")
    plt.close(fig)

# ------------------------------------------------------------------ summary tables
def table_at(k):
    lines = [f"\n### Results at k = {k} bands (mean ± std over 5 runs)\n"]
    for n, t in DS:
        df = sweeps[n]
        kk = k if (df.k == k).any() else df[df.method != "Full bands"].k.min()
        lines.append(f"\n**{t}** (k = {kk})\n\n| Method | SAD ↓ | RMSE ↓ | RE ↓ |\n|---|---|---|---|")
        sub = pd.concat([df[df.method == "Full bands"], df[(df.k == kk) & (df.method != "Full bands")]])
        g = sub.groupby("method")[["SAD", "RMSE", "RE"]].agg(["mean", "std"])
        best = {c: g[(c, "mean")].drop("Full bands").min() for c in ["SAD", "RMSE", "RE"]}
        for m in ["Full bands"] + ORDER:
            cells = []
            for c in ["SAD", "RMSE", "RE"]:
                mu, sd = g.loc[m, (c, "mean")], g.loc[m, (c, "std")]
                if np.isnan(mu):
                    cells.append("–")
                    continue
                s = f"{mu:.4f} ± {sd:.4f}"
                cells.append(f"**{s}**" if m != "Full bands" and np.isclose(mu, best[c]) else s)
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return lines

md += ["# Results tables"] + table_at(15)

# average rank across datasets and k (lower = better)
rank_rows = []
for n, t in DS:
    df = sweeps[n][sweeps[n].method != "Full bands"]
    for metric in ["SAD", "RMSE", "RE"]:
        if df[metric].isna().all():
            continue
        g = df.groupby(["k", "method"])[metric].mean().unstack()
        rk = g.rank(axis=1).mean()
        for m, v in rk.items():
            rank_rows.append(dict(dataset=t, metric=metric, method=m, rank=v))
rk = pd.DataFrame(rank_rows)
piv = rk.pivot_table(index="method", columns="metric", values="rank").loc[ORDER]
piv["Overall"] = piv.mean(axis=1)
piv = piv.sort_values("Overall")
md += ["\n### Average rank over all datasets and all k (1 = best of 9)\n", piv.round(2).to_markdown()]
piv.round(2).to_csv(f"{R}/avg_rank.csv")

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.barh(piv.index[::-1], piv["Overall"][::-1], color=[COL[m] for m in piv.index[::-1]])
ax.set_xlabel("Average rank (lower is better)")
ax.set_title("Overall ranking across datasets, k and metrics")
fig.savefig(f"{F}/avg_rank.png")
plt.close(fig)

# area-under-curve style: mean over k
md.append("\n### Mean over k = 5…30 (Cuprite: 15…30)\n")
rows = []
for n, t in DS:
    df = sweeps[n]
    full = df[df.method == "Full bands"][["SAD", "RMSE", "RE"]].mean()
    g = df[df.method != "Full bands"].groupby("method")[["SAD", "RMSE", "RE"]].mean()
    for m in ORDER:
        rows.append(dict(Dataset=t, Method=m, **g.loc[m].round(4).to_dict()))
    rows.append(dict(Dataset=t, Method="Full bands", **full.round(4).to_dict()))
md.append(pd.DataFrame(rows).to_markdown(index=False))

# ------------------------------------------------------------------ correlation study
corr_rows = []
for n, t in DS:
    c = pd.read_csv(f"{R}/corr_{n}.csv")
    for f in ["NMI", "SSIM", "VI", "Entropy", "GSR", "Spread"]:
        for metric in ["SAD", "RMSE", "RE"]:
            if c[metric].isna().all():
                continue
            rho, pv = spearmanr(c[f], c[metric])
            corr_rows.append(dict(dataset=t, feature=f, metric=metric, rho=rho, p=pv))
cr = pd.DataFrame(corr_rows)
cr.to_csv(f"{R}/spearman.csv", index=False)
feats = ["NMI", "SSIM", "VI", "Entropy", "GSR", "Spread"]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, metric in zip(axs, ["SAD", "RMSE", "RE"]):
    sub = cr[cr.metric == metric]
    dss = sub.dataset.unique()
    w = 0.8 / len(dss)
    for i, t in enumerate(dss):
        v = sub[sub.dataset == t].set_index("feature").loc[feats]
        ax.bar(np.arange(len(feats)) + i * w - 0.4 + w / 2, v["rho"], w, label=t)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(["mean NMI", "mean SSIM", "mean VI", "mean H", "GSR reach", "band spread"],
                       rotation=30, fontsize=8)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_title(f"Spearman ρ with {metric}")
axs[0].set_ylabel("ρ  (negative = higher feature → lower error)")
axs[-1].legend(fontsize=8)
fig.suptitle("Which subset property predicts unmixing error? (80 random 15-band subsets / dataset)", y=1.03)
fig.savefig(f"{F}/spearman.png")
plt.close(fig)
pv = cr.pivot_table(index="feature", columns=["dataset", "metric"], values="rho").loc[feats].round(2)
md += ["\n### Spearman correlation: subset property vs unmixing error (k = 15, 80 random subsets)\n",
       pv.to_markdown()]

# ------------------------------------------------------------------ ablation
md.append("\n### Ablation: endmember extractor (k = 15)\n")
for n, t in DS:
    a = pd.read_csv(f"{R}/ablation_{n}.csv")
    p = a.pivot_table(index="method", columns="extractor", values=["SAD", "RMSE"]).loc[ORDER].round(4)
    p = p.dropna(axis=1, how="all")
    md += [f"\n**{t}**\n", p.to_markdown()]

# ------------------------------------------------------------------ runtime
md.append("\n### Runtime (seconds, mean per run)\n")
trow = []
for n, t in DS:
    df = sweeps[n]
    bj = json.load(open(f"{R}/bands_{n}.json"))
    full = df[df.method == "Full bands"].t_unmix.mean()
    kk = 15
    sub = df[df.k == kk].groupby("method")[["t_bs", "t_unmix"]].mean()
    for m in ORDER:
        trow.append(dict(Dataset=t, Method=m, BS_time=round(sub.loc[m, "t_bs"], 3),
                         Unmix_time_k15=round(sub.loc[m, "t_unmix"], 3), Unmix_time_full=round(full, 3)))
    trow.append(dict(Dataset=t, Method="(similarity matrices, one-off)", BS_time=round(bj["t_sim"], 1)))
md.append(pd.DataFrame(trow).to_markdown(index=False))

open(f"{R}/tables.md", "w").write("\n".join(md))

# ------------------------------------------------------------------ qualitative figures
for n, t in DS:
    d = data.load(n)
    bj = json.load(open(f"{R}/bands_{n}.json"))["bands"]
    B = d["Y"].shape[0]
    wl = np.arange(B)
    # selected bands over the ground-truth endmember spectra
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 6.2), gridspec_kw=dict(height_ratios=[1.1, 1.4]))
    for j in range(min(d["p"], 6)):
        a0.plot(wl, d["M"][:, j], label=d["names"][j])
    a0.set_title(f"{t}: ground-truth endmember spectra")
    a0.set_xlim(0, B - 1)
    a0.legend(fontsize=7, ncol=6)
    for i, m in enumerate(ORDER):
        b = bj[m]["15"]
        a1.scatter(b, np.full(len(b), i), marker="|", s=180, color=COL[m])
    a1.set_yticks(range(len(ORDER)))
    a1.set_yticklabels(ORDER)
    a1.set_xlim(0, B - 1)
    a1.set_xlabel("Band index")
    a1.set_title("Bands chosen by each method (k = 15)")
    fig.savefig(f"{F}/bands_{n}.png")
    plt.close(fig)

    # abundance maps: GT | full | SR-NMI-VI | GSR-GA-U  (k=15)
    if d["A"] is not None:
        cfgs = [("Ground truth", None), ("Full bands", np.arange(B)),
                ("SR-NMI-VI (15)", bj["SR-NMI-VI"]["15"]), ("GSR-GA-U (15)", bj["GSR-GA-U"]["15"]),
                ("MaxVar (15)", bj["MaxVar"]["15"])]
        p = d["p"]
        fig, axs = plt.subplots(p, len(cfgs), figsize=(2.1 * len(cfgs), 2.0 * p))
        for c, (lab, b) in enumerate(cfgs):
            if b is None:
                Ahat, ttl = d["A"], lab
            else:
                o = unmix.unmix_pipeline(d["Y"], np.array(b), p, d["M"], d["A"], seed=0)
                Ahat, ttl = o["A"], f"{lab}\nRMSE {o['RMSE']:.3f}"
            for r in range(p):
                axs[r, c].imshow(data.to_image(Ahat[r], d["rows"], d["cols"]), cmap="jet", vmin=0, vmax=1)
                axs[r, c].set_xticks([]); axs[r, c].set_yticks([]); axs[r, c].grid(False)
                if c == 0:
                    axs[r, c].set_ylabel(d["names"][r])
            axs[0, c].set_title(ttl, fontsize=8.5)
        fig.suptitle(f"{t}: abundance maps", y=1.0)
        fig.savefig(f"{F}/abund_{n}.png")
        plt.close(fig)

    # estimated vs true endmembers (GSR-GA-U, k=15)
    o = unmix.unmix_pipeline(d["Y"], np.array(bj["GSR-GA-U"]["15"]), d["p"], d["M"], d["A"], seed=0)
    q = min(d["p"], 6)
    fig, axs = plt.subplots(1, q, figsize=(2.6 * q, 2.4))
    for j in range(q):
        mt = d["M"][:, j] / np.linalg.norm(d["M"][:, j]); me = o["E"][:, j] / np.linalg.norm(o["E"][:, j])
        axs[j].plot(mt, "k", label="GT"); axs[j].plot(me, "r--", label="estimated")
        axs[j].set_title(f"{d['names'][j]}\nSAD {o['SAD_each'][j]:.3f}", fontsize=8.5)
        axs[j].set_xticks([]); axs[j].set_yticks([])
    axs[0].legend(fontsize=7)
    fig.suptitle(f"{t}: endmembers from 15 GSR-GA-U bands vs ground truth", y=1.08)
    fig.savefig(f"{F}/endmembers_{n}.png")
    plt.close(fig)

    # similarity matrices
    s = similarity.get_all(d)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (k_, ttl) in zip(axs, [("NMI", "NMI (shared information)"), ("SSIM", "SSIM (structural similarity)"),
                                   ("VIn", "VI (normalised, dissimilarity)")]):
        im = ax.imshow(s[k_], cmap="viridis"); ax.set_title(ttl); ax.grid(False)
        ax.set_xlabel("band"); plt.colorbar(im, ax=ax, fraction=0.046)
    axs[0].set_ylabel("band")
    fig.suptitle(f"{t}: band-to-band similarity matrices", y=1.02)
    fig.savefig(f"{F}/similarity_{n}.png")
    plt.close(fig)

# GA convergence + knee point on best-RE-vs-k curve
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for n, t in DS:
    h = json.load(open(f"{R}/ga_hist_{n}.json"))
    axs[0].plot(np.array(h) / h[0], "-o", ms=3, label=t)
    df = sweeps[n]
    g = df[df.method == "GSR-GA-U"].groupby("k")["RE"].mean()
    y = (g.values - g.values.min()) / (g.values.max() - g.values.min() + 1e-12)
    x = (g.index.values - g.index.min()) / (g.index.max() - g.index.min())
    knee = g.index.values[np.argmax((1 - x) - y)] if len(g) > 2 else g.index[0]
    axs[1].plot(g.index, g.values, "-o", label=f"{t} (knee k={knee})")
    axs[1].plot(knee, g.loc[knee], "r*", ms=13)
axs[0].set_title("GSR-GA-U convergence (k=15)"); axs[0].set_xlabel("generation")
axs[0].set_ylabel("best fitness / initial"); axs[0].legend(fontsize=8)
axs[1].set_title("Trade-off: reconstruction error vs |S|"); axs[1].set_xlabel("k"); axs[1].set_ylabel("RE")
axs[1].legend(fontsize=8)
fig.savefig(f"{F}/ga_pareto.png")
plt.close(fig)

# pipeline diagram
fig, ax = plt.subplots(figsize=(13, 1.6)); ax.axis("off")
steps = ["Hyperspectral cube\nY (B bands × N px)", "Band selection\n(9 methods → k bands)", "VCA on k bands\n→ pure-pixel indices",
         "FCLS on k bands\n→ abundances A", "Full-band spectra at\nthose pixels → E", "Evaluate\nSAD · RMSE · RE"]
X = np.arange(len(steps)) * 1.3
for i, s_ in enumerate(steps):
    ax.text(X[i], 0.5, s_, ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fde7d6" if i == 1 else "#e8f0fb", ec="#345"))
    if i < len(steps) - 1:
        ax.annotate("", xy=(X[i + 1] - 0.56, 0.5), xytext=(X[i] + 0.56, 0.5), arrowprops=dict(arrowstyle="-|>", lw=1.6))
ax.set_xlim(-0.65, X[-1] + 0.65); ax.set_ylim(0.2, 0.8)
fig.savefig(f"{F}/pipeline.png")
plt.close(fig)
print("figures done")
