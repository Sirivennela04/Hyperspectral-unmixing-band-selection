"""Interactive demo:  streamlit run app.py

Pick a dataset, a band-selection method and k -> see the selected bands, the estimated
endmembers and abundance maps, and SAD / RMSE / reconstruction error against full bands.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from hsu import data, similarity, unmix
from hsu import band_selection as bs

st.set_page_config(page_title="Band Selection for Hyperspectral Unmixing", layout="wide")
st.title("Band Selection for Hyperspectral Unmixing — live demo")
st.caption("Pipeline: band selection → VCA / N-FINDR endmember extraction → FCLS abundances → SAD · RMSE · RE")


@st.cache_resource
def get(ds):
    d = data.load(ds)
    return d, similarity.get_all(d)


@st.cache_data
def run(ds, bands, extractor, seed):
    d, _ = get(ds)
    t = time.time()
    o = unmix.unmix_pipeline(d["Y"], np.array(bands), d["p"], d["M"], d["A"], seed=seed, extractor=extractor)
    o["time"] = time.time() - t
    return o


with st.sidebar:
    ds = st.selectbox("Dataset", ["samson", "jasper", "urban", "cuprite"],
                      format_func=lambda s: {"samson": "Samson", "jasper": "Jasper Ridge",
                                             "urban": "Urban", "cuprite": "Cuprite"}[s])
    d, sim = get(ds)
    B = d["Y"].shape[0]
    method = st.selectbox("Band-selection method", list(bs.METHODS), index=6)
    k = st.slider("Number of bands k", max(5, d["p"] + 1), 40, 15)
    extractor = st.radio("Endmember extractor", ["vca", "nfindr"], format_func=str.upper)
    seed = st.number_input("Random seed", 0, 99, 0)
    go = st.button("Run unmixing", type="primary")

st.write(f"**{d['name']}** — {d['rows']}×{d['cols']} pixels, **{B} bands**, {d['p']} endmembers: "
         + ", ".join(d["names"]))

if go or "last" in st.session_state:
    if go:
        t = time.time()
        with st.spinner(f"Selecting {k} bands with {method}…"):
            bands = bs.select(method, d, sim, k, seed=seed)
        tbs = time.time() - t
        st.session_state["last"] = (ds, method, k, extractor, seed, [int(b) for b in bands], tbs)
    # r_* = settings of the last "Run unmixing" click (what the results on screen show).
    # method / k / extractor / seed keep the CURRENT sidebar values (used by the comparison).
    ds_, r_method, r_k, r_ext, r_seed, bands, tbs = st.session_state["last"]
    if ds_ != ds:
        st.info("Dataset changed — press **Run unmixing**.")
        st.stop()
    if (r_method, r_k, r_ext, r_seed) != (method, k, extractor, seed):
        st.warning(f"The sidebar has changed. The results below are still for **{r_method}, k = {r_k}, "
                   f"{r_ext.upper()}, seed {r_seed}**. Press **Run unmixing** to update them.")
    o = run(ds, tuple(bands), r_ext, r_seed)
    of = run(ds, tuple(range(B)), r_ext, r_seed)

    c = st.columns(5)
    c[0].metric("Bands used", f"{len(bands)} / {B}", f"-{100 * (1 - len(bands) / B):.0f}% data")
    c[1].metric("SAD (rad)", f"{o['SAD']:.4f}", f"{o['SAD'] - of['SAD']:+.4f} vs full", delta_color="inverse")
    if "RMSE" in o:
        c[2].metric("Abundance RMSE", f"{o['RMSE']:.4f}", f"{o['RMSE'] - of['RMSE']:+.4f} vs full",
                    delta_color="inverse")
    else:
        c[2].metric("Abundance RMSE", "n/a", "no GT abundances")
    c[3].metric("Reconstruction err", f"{o['RE']:.4f}", f"{o['RE'] - of['RE']:+.4f} vs full", delta_color="inverse")
    c[4].metric("Unmixing time", f"{o['time']:.2f}s", f"full: {of['time']:.2f}s", delta_color="off")
    st.caption(f"Band selection took {tbs:.2f}s. Selected bands: {bands}")

    tab1, tab2, tab3, tab4 = st.tabs(["Selected bands", "Endmembers", "Abundance maps", "Similarity matrices"])
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        mean_sp = d["Y"].mean(axis=1)
        ax.plot(mean_sp, "k", lw=1, label="mean pixel spectrum")
        for j in range(min(d["p"], 6)):
            ax.plot(d["M"][:, j] / d["M"][:, j].max() * mean_sp.max(), lw=0.8, alpha=0.6, label=d["names"][j])
        for b in bands:
            ax.axvline(b, color="r", alpha=0.6, lw=1.3)
        ax.set_xlabel("band index"); ax.legend(fontsize=7, ncol=4); ax.set_xlim(0, B - 1)
        ax.set_title(f"{r_method}: {len(bands)} selected bands (red lines)")
        st.pyplot(fig)
    with tab2:
        q = min(d["p"], 6)
        fig, axs = plt.subplots(1, q, figsize=(2.7 * q, 2.4))
        for j in range(q):
            axs[j].plot(d["M"][:, j] / np.linalg.norm(d["M"][:, j]), "k", label="ground truth")
            axs[j].plot(o["E"][:, j] / np.linalg.norm(o["E"][:, j]), "r--", label="estimated")
            axs[j].set_title(f"{d['names'][j]}  SAD={o['SAD_each'][j]:.3f}", fontsize=9)
            axs[j].set_xticks([]); axs[j].set_yticks([])
        axs[0].legend(fontsize=7)
        st.pyplot(fig)
    with tab3:
        if d["A"] is None:
            st.info("Cuprite has no ground-truth abundances; showing estimated maps only.")
        p = d["p"] if d["A"] is not None else min(d["p"], 6)
        rows = [("Ground truth", d["A"])] if d["A"] is not None else []
        rows += [("Full bands", of["A"]), (f"{r_method} ({len(bands)} bands)", o["A"])]
        fig, axs = plt.subplots(len(rows), p, figsize=(2.2 * p, 2.2 * len(rows)))
        axs = np.atleast_2d(axs)
        for r, (lab, A) in enumerate(rows):
            for j in range(p):
                axs[r, j].imshow(data.to_image(A[j], d["rows"], d["cols"]), cmap="jet", vmin=0, vmax=1)
                axs[r, j].set_xticks([]); axs[r, j].set_yticks([])
                if r == 0:
                    axs[r, j].set_title(d["names"][j], fontsize=9)
            axs[r, 0].set_ylabel(lab, fontsize=8)
        st.pyplot(fig)
    with tab4:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, (kk, t) in zip(axs, [("NMI", "NMI"), ("SSIM", "SSIM"), ("VIn", "VI (normalised)")]):
            ax.imshow(sim[kk], cmap="viridis"); ax.set_title(t)
            ax.scatter(bands, bands, c="r", s=8)
        st.pyplot(fig)
        st.caption("Red dots: selected bands on the diagonal. Bright blocks = groups of redundant neighbouring bands.")

    with st.expander("Compare all methods at this k"):
        st.caption("VCA uses random directions, so a single run is noisy. Compare methods on the mean of several seeds.")
        t_bench, t_live = st.tabs(["Benchmark averages (5 runs, precomputed)", "Live comparison"])
        with t_bench:
            fpath = f"results/sweep_{ds}.csv"
            try:
                sw = pd.read_csv(fpath)
                ks = sorted(sw[sw.method != "Full bands"].k.unique())
                kb = k if k in ks else min(ks, key=lambda x: abs(x - k))
                if kb != k:
                    st.info(f"The benchmark has k = {', '.join(map(str, ks))}; showing k = {kb}.")
                sub = pd.concat([sw[(sw.k == kb) & (sw.method != "Full bands")], sw[sw.method == "Full bands"]])
                g = sub.groupby("method")[["SAD", "RMSE", "RE"]].agg(["mean", "std"])
                order = list(bs.METHODS) + ["Full bands"]
                g = g.loc[[m for m in order if m in g.index]]
                means = g.xs("mean", axis=1, level=1).dropna(axis=1, how="all")
                stds = g.xs("std", axis=1, level=1)[means.columns]
                show = means.copy().astype(object)
                for c_ in means.columns:
                    show[c_] = [f"{mu:.4f} ± {sd:.4f}" for mu, sd in zip(means[c_], stds[c_])]
                best = {c_: means[c_].drop("Full bands", errors="ignore").idxmin() for c_ in means.columns}
                st.dataframe(show.style.apply(lambda col: ["background-color: #2e7d32; color: white" if idx == best[col.name] else "" for idx in col.index], axis=0))
                st.caption(f"k = {kb}, mean ± std of 5 runs (seeds 0–4) from run_experiments.py. Green = best band-selection method per column. Lower is better.")
            except FileNotFoundError:
                st.info("Run run_experiments.py first to create the benchmark results.")
        with t_live:
            nruns = st.slider("Number of seeds to average", 1, 5, 3, help="Seeds start at the sidebar seed.")
            st.caption(f"Uses the current sidebar settings: {ds}, k = {k}, {extractor.upper()}, seeds {seed}–{seed + nruns - 1}. "
                       "Deterministic methods pick the same bands for every seed; Random and GSR-GA-U pick new bands per seed.")
            if st.button("Run comparison"):
                res, bar = [], st.progress(0.0)
                ms = list(bs.METHODS)
                for i_, m in enumerate(ms):
                    vals, first_bands = [], None
                    for s_ in range(seed, seed + nruns):
                        b = bs.select(m, d, sim, k, seed=s_)
                        first_bands = first_bands if first_bands is not None else [int(x) for x in b]
                        r = run(ds, tuple(int(x) for x in b), extractor, s_)
                        vals.append((r["SAD"], r.get("RMSE", np.nan), r["RE"]))
                    v = np.array(vals, dtype=float)
                    res.append(dict(method=m, SAD=v[:, 0].mean(), SAD_std=v[:, 0].std(), RMSE=v[:, 1].mean(), RMSE_std=v[:, 1].std(),
                                    RE=v[:, 2].mean(), RE_std=v[:, 2].std(), bands=str(first_bands)))
                    bar.progress((i_ + 1) / (len(ms) + 1))
                vf = np.array([(lambda r: (r["SAD"], r.get("RMSE", np.nan), r["RE"]))(run(ds, tuple(range(B)), extractor, s_)) for s_ in range(seed, seed + nruns)], dtype=float)
                res.append(dict(method="Full bands", SAD=vf[:, 0].mean(), SAD_std=vf[:, 0].std(), RMSE=vf[:, 1].mean(), RMSE_std=vf[:, 1].std(),
                                RE=vf[:, 2].mean(), RE_std=vf[:, 2].std(), bands="all"))
                bar.progress(1.0)
                df = pd.DataFrame(res).set_index("method")
                show = pd.DataFrame(index=df.index)
                for c_ in ["SAD", "RMSE", "RE"]:
                    if df[c_].notna().any():
                        show[c_] = [f"{mu:.4f} ± {sd:.4f}" for mu, sd in zip(df[c_], df[c_ + "_std"])]
                show["bands (first seed)"] = df["bands"]
                best = {c_: df[c_].drop("Full bands").idxmin() for c_ in ["SAD", "RMSE", "RE"] if df[c_].notna().any()}
                st.dataframe(show.style.apply(lambda col: ["background-color: #2e7d32; color: white" if col.name in best and idx == best[col.name] else "" for idx in col.index], axis=0))
                st.caption("Mean ± std over the chosen seeds. Green = best band-selection method per column. Lower is better.")
else:
    st.info("Choose settings in the sidebar and press **Run unmixing**.")
