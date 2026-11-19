"""Dataset loaders for hyperspectral unmixing benchmarks.

Every loader returns a dict with
    name   : dataset name
    Y      : (B, N) float64 reflectance matrix, scaled to roughly [0, 1]
    rows, cols : image size (N = rows * cols, MATLAB column-major pixel order)
    M      : (B, p) ground-truth endmember spectra (or None)
    A      : (p, N) ground-truth abundances (or None, e.g. Cuprite)
    p      : number of endmembers
    names  : endmember names
"""
import os
import numpy as np
import scipy.io as sio

DATA_DIR = os.environ.get("HSU_DATA", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))


def _p(*a):
    return os.path.join(DATA_DIR, *a)


def load_samson():
    d = sio.loadmat(_p("Samson", "samson_1.mat"))
    g = sio.loadmat(_p("Samson", "end3.mat"))
    Y = d["V"].astype(float)
    return dict(name="Samson", Y=Y, rows=95, cols=95, M=g["M"].astype(float), A=g["A"].astype(float),
                p=3, names=["Soil", "Tree", "Water"])


def load_jasper():
    d = sio.loadmat(_p("Jasper_Ridge", "jasperRidge2_R198.mat"))
    g = sio.loadmat(_p("Jasper_Ridge", "end4.mat"))
    Y = d["Y"].astype(float) / float(d["maxValue"].squeeze())
    return dict(name="Jasper Ridge", Y=Y, rows=100, cols=100, M=g["M"].astype(float), A=g["A"].astype(float),
                p=4, names=["Tree", "Water", "Soil", "Road"])


def load_urban(nend=4):
    d = sio.loadmat(_p("Urban", "Urban_R162.mat"))
    g = sio.loadmat(_p("Urban", f"end{nend}_groundTruth.mat"))
    Y = d["Y"].astype(float) / float(d["maxValue"].squeeze())
    names = {4: ["Asphalt", "Grass", "Tree", "Roof"],
             5: ["Asphalt", "Grass", "Tree", "Roof", "Dirt"],
             6: ["Asphalt", "Grass", "Tree", "Roof", "Metal", "Dirt"]}[nend]
    return dict(name=f"Urban", Y=Y, rows=307, cols=307, M=g["M"].astype(float), A=g["A"].astype(float),
                p=nend, names=names)


def load_cuprite():
    d = sio.loadmat(_p("Cuprite", "CupriteS1_R188.mat"))
    g = sio.loadmat(_p("Cuprite", "groundTruth_Cuprite_nEnd12.mat"))
    Y = d["Y"].astype(float)
    Y = np.clip(Y / 10000.0, 0, 1)           # reflectance x 10^4 ; clip a few corrupt values
    sel = g["slctBnds"].squeeze().astype(int) - 1   # MATLAB 1-based -> 0-based
    M = g["M"][sel].astype(float)
    names = ["Alunite", "Andradite", "Buddingtonite", "Dumortierite", "Kaolinite 1", "Kaolinite 2",
             "Muscovite", "Montmorillonite", "Nontronite", "Pyrope", "Sphene", "Chalcedony"]
    return dict(name="Cuprite", Y=Y, rows=250, cols=190, M=M, A=None, p=12, names=names)


LOADERS = {"samson": load_samson, "jasper": load_jasper, "urban": load_urban, "cuprite": load_cuprite}


def load(name):
    return LOADERS[name.lower()]()


def to_image(vec_or_mat, rows, cols):
    """(N,) or (C, N) column-major pixel vector -> (rows, cols) or (C, rows, cols) image."""
    v = np.asarray(vec_or_mat)
    if v.ndim == 1:
        return v.reshape((rows, cols), order="F")
    return v.reshape((v.shape[0], rows, cols), order="F")
