"""Command-line demo.

python demo.py --dataset samson --method SR-NMI-VI --k 15
python demo.py --dataset jasper --all --k 10        # compare all methods
"""
import argparse
import time

import numpy as np

from hsu import data, similarity, unmix
from hsu import band_selection as bs

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="samson", choices=list(data.LOADERS))
ap.add_argument("--method", default="SR-NMI-VI", choices=list(bs.METHODS))
ap.add_argument("--k", type=int, default=15)
ap.add_argument("--extractor", default="vca", choices=["vca", "nfindr"])
ap.add_argument("--all", action="store_true", help="compare all methods")
a = ap.parse_args()

d = data.load(a.dataset)
B = d["Y"].shape[0]
print(f"\nDataset: {d['name']}  |  {d['rows']}x{d['cols']} px, {B} bands, {d['p']} endmembers {d['names']}")
sim = similarity.get_all(d)


def show(name, bands):
    t = time.time()
    o = unmix.unmix_pipeline(d["Y"], bands, d["p"], d["M"], d["A"], extractor=a.extractor)
    rm = f"{o['RMSE']:.4f}" if "RMSE" in o else "  n/a "
    print(f"{name:<12} k={len(bands):>3}  SAD={o['SAD']:.4f}  RMSE={rm}  RE={o['RE']:.4f}  ({time.time() - t:.2f}s)")
    return o


print("-" * 78)
show("Full bands", np.arange(B))
for m in (bs.METHODS if a.all else [a.method]):
    b = bs.select(m, d, sim, a.k)
    o = show(m, b)
    if not a.all:
        print("selected bands:", list(map(int, b)))
        print("per-endmember SAD:", dict(zip(d["names"], np.round(o["SAD_each"], 4))))
