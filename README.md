# flare-forecast

Forecasting of solar flare counts (C-, M-, and X-class) within the next 24 hours from active-region magnetogram cutouts acquired by SOHO/MDI and SDO/HMI. The repository implements both count forecasting through Poisson regression and flare-occurrence forecasting through binary classification, using convolutional neural networks evaluated with 5-fold cross-validation repeated across multiple random seeds.

## Forecast targets

For each active-region magnetogram, the models predict either:

### Poisson regression

Expected flare counts

- λC: expected number of C-class flares
- λM: expected number of M-class flares
- λX: expected number of X-class flares

### Binary classification

Flare occurrence probabilities

- P(C ≥ 1)
- P(M ≥ 1)
- P(X ≥ 1)

Forecast labels are generated using NOAA/SWPC flare reports retrieved from HEK and counted within a 24-hour forecast window.

---


## Repository structure

```text
flare-forecast/
├── configs/
│   ├── default.yml
│   └── smoke.yml
├── data/                     # full dataset (not tracked)
├── notebooks/                # result visualisation
├── results/                  # metrics and predictions
├── scripts/
│   ├── preprocessing/
│   │   ├── 01_organize_fits.py
│   │   ├── 02_index_clean_magnetograms.py
│   │   └── 03_label_flare_number.py
│   ├── make_subset.py
│   └── train_kfold_multiseed.py
├── src/flare_forecast/
│   ├── analysis/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
│   └── data/subset/
└── pyproject.toml
```

### Package overview

| Module | Purpose |
|----------|----------|
| `utils/` | FITS handling, calibration, SunPy geometry, magnetic feature extraction |
| `data/` | datasets, train/validation/test splits, augmentations |
| `models/` | Poisson and binary CNN architectures |
| `training/` | model construction, calibration, callbacks, training routines |
| `analysis/` | aggregation of cross-validation results |

---

## Installation

Python >= 3.10 (developed and tested with Python 3.12).

```bash
git clone <repo-url>
cd flare-forecast
pip install -e .
```

Main dependencies:

```text
torch
torchvision
pytorch-lightning
torchmetrics
numpy
pandas
scipy
scikit-learn
h5py
pyyaml
```

Additional preprocessing dependencies:

```text
sunpy
astropy
scikit-image
```

Optional:

```text
comet-ml
```

Disable Comet logging by setting:

```yaml
comet:
  enabled: false
```

All commands should be run from the repository root.

---

## Data

The full dataset is not distributed with this repository.

Experiments are based on the public ARCAFF:CCD dataset:

https://doi.org/10.5281/zenodo.17865447

### Dataset preparation

1. Download ARCAFF:CCD.
2. Run the preprocessing pipeline.

#### Step 1 — Organize FITS files

```bash
CSV_PATH=<region_catalogue.csv> \
DEST_BASE=data \
python scripts/preprocessing/01_organize_fits.py
```

Creates:

```text
data/
├── MDI/<year>/
└── HMI/<year>/
```

#### Step 2 — Index and clean magnetograms

```bash
python scripts/preprocessing/02_index_clean_magnetograms.py \
    MDI HMI \
    --root <fits_root> \
    --newdir <hdf5_dir> \
    --indexdir <index_dir>
```

This stage:

- performs quality control
- calibrates magnetograms
- computes magnetic flux quantities
- converts FITS files to HDF5
- generates index files

Outputs:

```text
index_MDI.csv
index_HMI.csv
index_MDI_HMI.csv
```

#### Step 3 — Generate flare labels

```bash
python scripts/preprocessing/03_label_flare_number.py \
    24 \
    <index_dir>/index_MDI_HMI.csv \
    <output>.csv
```

Labels are generated from HEK flare events reported by SWPC.

Output includes:

```text
C_count
M_count
X_count
```

plus flare timing information and scalar magnetic features.

### Configure the dataset

Point

```yaml
data:
  data_file:
```

in `configs/default.yml` to the generated labelled CSV.

The dataset must contain at least:

```text
filename
sample_time
ar_number
C_count
M_count
X_count
```

Optional scalar features include:

```text
tot_us_flux
tot_flux
datamin
datamax
schrijver_R
wlsg
grad_mean
grad_max
bz_skew
bz_kurt
```

---

## Test subset

A small demonstration subset is provided:

```text
tests/data/subset/
```

It contains:

- CSV labels
- HDF5 magnetograms

and allows the complete pipeline to run without downloading the full dataset.

Results obtained on this subset are not scientifically meaningful.

To regenerate the subset:

```bash
python scripts/make_subset.py \
    --csv <labelled.csv> \
    --base <dataset_root> \
    --out tests/data/subset
```

---

## Usage

### Quick smoke test

```bash
python scripts/train_kfold_multiseed.py \
    --config configs/smoke.yml \
    --seeds 0 \
    --folds 0
```

### Full cross-validation experiment

```bash
python scripts/train_kfold_multiseed.py \
    --config configs/default.yml
```

Example:

```bash
python scripts/train_kfold_multiseed.py \
    --config configs/default.yml \
    --seeds 0 1 2 \
    --modes poisson \
    --folds 0 1
```


## Single-run training

For a single train/validation/test experiment with Comet logging:

```bash
python -m flare_forecast.training.train \
    --config configs/default.yml \
    --mode poisson
```

or

```bash
python -m flare_forecast.training.train \
    --config configs/default.yml \
    --mode binary
```

---

## Results

Outputs are written to:

```text
results/
```


### Predictions

```text
predictions_<tag>_seed_<s>.csv
```

Contains sample-level predictions and ground-truth values.

### Aggregation

Merge all folds and seeds:

```bash
python -m flare_forecast.analysis.merge_results \
    --tag val_TSS_C_feat-NONE
```

Output:

```text
kfold_all_seeds_metrics_<tag>.csv
kfold_all_seeds_predictions_<tag>.csv
```

---


### Visualise published results

The notebook in `notebooks/` reproduces and visualises the original experimental results.

---

## Data citation

If you use this repository, please cite the ARCAFF:CCD dataset:

```text
ARCAFF:CCD — Active Region Cutout Archive for Flare Forecasting
https://doi.org/10.5281/zenodo.17865447
```
