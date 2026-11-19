# Band Selection for Hyperspectral Unmixing: a comparative study

This project tests whether a small subset of spectral bands, picked by a band-selection (BS)
algorithm, can keep unmixing accuracy close to what all bands give.

**Fixed pipeline (only the bands change):**
`HSI cube → band selection (k bands) → VCA (or N-FINDR) endmembers → FCLS abundances → SAD / RMSE / RE`

| | |
|---|---|
| Datasets | Samson (156 b, 3 EM), Jasper Ridge (198 b, 4 EM), Urban (162 b, 4 EM), Cuprite (188 b, 12 EM, endmember GT only) |
| BS methods | Uniform, Random, MaxVar, IE (entropy), OPBS, WaLuMI, **SR-NMI-VI**, **GSR-Hub** (MO-GSR-BS phase 1), **GSR-GA-U** (proposed unmixing-aware GA) |
| Unmixing | VCA + FCLS (N-FINDR as ablation) |
| Metrics | SAD (endmembers), RMSE (abundances), RE (reconstruction) |
| Sweep | k = 5, 10, 15, 20, 25, 30 (Cuprite 15–30), 5 runs each |

## Setup
```bash
pip install -r requirements.txt
```
The `.mat` files go in `data/<Dataset>/` (included in the zip). Source: the
`cu4oral6/hsi_datasets_unmixing` GitHub mirror of the standard benchmark files
(Zhu et al., rslab.ut.ac.ir).

## Run
```bash
python demo.py --dataset samson --method SR-NMI-VI --k 15     # one method
python demo.py --dataset jasper --all --k 10                  # all methods
streamlit run app.py                                          # interactive demo (browser)
python run_experiments.py samson jasper urban cuprite         # full benchmark (~40 min)
python make_figures.py                                        # tables + figures
```

## Code layout
```
hsu/data.py            dataset loaders
hsu/unmix.py           VCA, N-FINDR, FCLS (batched FISTA), SAD/RMSE/RE, Hungarian matching
hsu/similarity.py      entropy, NMI, VI, SSIM band-pair matrices (cached)
hsu/band_selection.py  9 BS algorithms
run_experiments.py     benchmark; make_figures.py  plots and tables
app.py / demo.py       demos
results/  figures/     outputs
```
