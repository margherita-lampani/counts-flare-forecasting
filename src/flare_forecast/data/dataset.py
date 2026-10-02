"""
PyTorch Lightning DataModule for multi-output solar flare count forecasting.

This module handles data loading, preprocessing, augmentation, feature
normalization, and train/validation/test splitting for active-region
magnetogram forecasting experiments. It supports joint prediction of
C-class, M-class, and X-class flare counts using magnetogram images and
optional scalar magnetic features.

Main features:
    - Multi-target flare-count prediction (C_count, M_count, X_count)
    - HDF5 magnetogram loading and normalization
    - Optional physics-derived scalar features
    - Configurable data augmentation strategies
    - Stratified active-region cross-validation without data leakage
    - Support for temporal and operational test scenarios
    - Automated feature standardization
    - Detailed split diagnostics and class-imbalance reporting

Supported split strategies:
    - Temporal holdout splits (test_a, test_b)
    - AR-based stratified k-fold cross-validation (test_c)
    - Hierarchical flare-level stratified k-fold validation (test_d)

Dataset outputs:
    filename
        Path to the magnetogram sample.

    image
        Normalized magnetogram tensor.

    features
        Optional standardized scalar feature vector.

    label
        Raw flare counts [C_count, M_count, X_count].

This module provides the data pipeline used by both Poisson regression
and binary-classification forecasting models.
"""

import sys, os

import torch
from torchvision import transforms
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

from flare_forecast.data.transforms import RandomPolaritySwitch


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — readable flare count summary per split
# ─────────────────────────────────────────────────────────────────────────────

def _print_flare_summary(name: str, df, target_cols: list):
    """
    Print how many positive/negative samples exist per class, mean λ,
    and max count — much more informative than describe() for understanding
    class imbalance.
    """
    n    = len(df)
    c0, c1, c2 = target_cols
    vals = {col: df[col].values for col in target_cols}
    pos  = {col: int((vals[col] >= 1).sum()) for col in target_cols}
    tot  = {col: int(vals[col].sum())        for col in target_cols}
    mean = {col: float(vals[col].mean())     for col in target_cols}
    mx   = {col: int(vals[col].max())        for col in target_cols}

    def fmt_pos(col):
        p = pos[col]; return f"{p:5d} ({100*p/n:3.0f}%)"

    def fmt_neg(col):
        k = n - pos[col]; return f"{k:5d} ({100*k/n:3.0f}%)"

    print(f"\n{'─'*56}")
    print(f"  {name}  (n={n})")
    print(f"  {'':14s}  {'C-class':>10}  {'M-class':>10}  {'X-class':>10}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*10}")
    print(f"  {'Pos (≥1)':14s}  {fmt_pos(c0):>10}  {fmt_pos(c1):>10}  {fmt_pos(c2):>10}")
    print(f"  {'Neg (=0)':14s}  {fmt_neg(c0):>10}  {fmt_neg(c1):>10}  {fmt_neg(c2):>10}")
    print(f"  {'Tot events':14s}  {tot[c0]:>10}  {tot[c1]:>10}  {tot[c2]:>10}")
    print(f"  {'Mean λ':14s}  {mean[c0]:>10.3f}  {mean[c1]:>10.3f}  {mean[c2]:>10.3f}")
    print(f"  {'Max count':14s}  {mx[c0]:>10}  {mx[c1]:>10}  {mx[c2]:>10}")
    print(f"{'─'*56}")


def split_data(df, val_split, test=''):
    """
    Split dataset into training, validation, hold-out (pseudotest) and test sets.

    Test set options:
        'test_a' : all data from November and December (cycle-agnostic)
        'test_b' : 2021-2022 (operational)
        'test_c' : Stratified K-Fold on ar_number using xrsb_max_in_* for stratification
        'test_d' : Stratified K-Fold on ar_number using hierarchical flare level
                   (1=C only, 2=at least 1 M, 3=at least 1 X).
        otherwise: test_a | test_b combined
    """

    # ------------------------------------------------------------------ #
    # TEST_D: STRATIFIED K-FOLD — HIERARCHICAL FLARE LEVEL                #
    # ------------------------------------------------------------------ #
    if test == 'test_d':

        c_col = next((c for c in df.columns if 'C_count' in c), None)
        m_col = next((c for c in df.columns if 'M_count' in c), None)
        x_col = next((c for c in df.columns if 'X_count' in c), None)

        if not all([c_col, m_col, x_col]):
            raise ValueError(
                "test_d requires C_count, M_count and X_count columns in the dataframe. "
                f"Found: {[c for c in [c_col, m_col, x_col] if c is not None]}"
            )

        ar_summary = (
            df.groupby('ar_number')
              .agg(**{
                  '_has_M': (m_col, lambda x: (x >= 1).any()),
                  '_has_X': (x_col, lambda x: (x >= 1).any()),
              })
              .reset_index()
        )

        ar_summary['_stratum'] = (
            1
            + ar_summary['_has_M'].astype(int)
            + ar_summary['_has_X'].astype(int)
        )

        strat_counts = ar_summary['_stratum'].value_counts().sort_index()
        labels_str = {1: 'C only', 2: 'M+', 3: 'X+'}
        print("\ntest_d stratum distribution across ARs:")
        for level, count in strat_counts.items():
            print(f"  stratum {level} ({labels_str[level]}): {count} ARs "
                  f"({100 * count / len(ar_summary):.1f}%)")

        N_SPLITS     = 5
        RANDOM_STATE = 42

        skf              = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                           random_state=RANDOM_STATE)
        fold_assignments = np.empty(len(ar_summary), dtype=int)
        for fold_idx, (_, fold_test_idx) in enumerate(
                skf.split(np.arange(len(ar_summary)),
                          ar_summary['_stratum'].values)):
            fold_assignments[fold_test_idx] = fold_idx
        ar_summary['_fold'] = fold_assignments

        ar_to_fold  = ar_summary.set_index('ar_number')['_fold'].to_dict()
        df          = df.copy()
        df['_fold'] = df['ar_number'].map(ar_to_fold)

        test_fold   = val_split % N_SPLITS
        buffer_fold = (val_split + 1) % N_SPLITS
        val_fold    = (val_split + 2) % N_SPLITS
        train_folds = [f for f in range(N_SPLITS)
                       if f not in (test_fold, buffer_fold, val_fold)]

        df_test = df[df['_fold'] == test_fold].drop(columns='_fold')
        df_val  = df[df['_fold'] == val_fold ].drop(columns='_fold')

        buffer_ars = ar_summary[ar_summary['_fold'] == buffer_fold].copy()

        rng           = np.random.default_rng(RANDOM_STATE)
        pseudo_ar_ids = []
        for stratum_val in sorted(buffer_ars['_stratum'].unique()):
            ars_in_stratum = buffer_ars.loc[
                buffer_ars['_stratum'] == stratum_val, 'ar_number'].values
            n_pseudo = max(1, int(round(len(ars_in_stratum) * 0.10)))
            chosen   = rng.choice(ars_in_stratum, size=n_pseudo, replace=False)
            pseudo_ar_ids.extend(chosen.tolist())

        df_buffer      = df[df['_fold'] == buffer_fold].drop(columns='_fold')
        df_pseudotest  = df_buffer[ df_buffer['ar_number'].isin(pseudo_ar_ids)]
        df_buffer_rest = df_buffer[~df_buffer['ar_number'].isin(pseudo_ar_ids)]

        df_train_folds = df[df['_fold'].isin(train_folds)].drop(columns='_fold')
        df_train       = pd.concat([df_train_folds, df_buffer_rest],
                                   ignore_index=True)

        ar_to_stratum = ar_summary.set_index('ar_number')['_stratum'].to_dict()
        print("\ntest_d stratum proportions per split (% of samples):")
        for name, split_df in [('Train', df_train), ('Val', df_val),
                                ('PseudoTest', df_pseudotest), ('Test', df_test)]:
            strata = split_df['ar_number'].map(ar_to_stratum)
            total  = len(split_df)
            parts  = '  '.join(
                f"s{lvl}={100 * (strata == lvl).sum() / total:.1f}%"
                for lvl in range(1, 4)
            )
            print(f"  {name:12s}  n={total:5d}  {parts}")

        return df_test, df_pseudotest, df_train, df_val

    # ------------------------------------------------------------------ #
    # TEST_C: STRATIFIED K-FOLD WITHOUT LEAKAGE ON AR NUMBERS             #
    # ------------------------------------------------------------------ #
    if test == 'test_c':

        candidates = [c for c in df.columns if c.startswith("xrsb_max_in_")]
        if not candidates:
            raise ValueError("No 'xrsb_max_in_*' column found in the dataframe.")
        label_col = candidates[0]

        N_SPLITS     = 5
        N_BINS       = 5
        RANDOM_STATE = 42

        ar_summary = (
            df.groupby("ar_number")[label_col]
            .max()
            .reset_index()
            .rename(columns={label_col: "_label_max"})
        )

        log_vals               = np.log10(np.clip(ar_summary["_label_max"].values, 1e-9, None))
        bin_edges              = np.linspace(log_vals.min(), log_vals.max(), N_BINS + 1)
        ar_summary["_stratum"] = np.digitize(log_vals, bin_edges[1:-1])

        skf              = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_assignments = np.empty(len(ar_summary), dtype=int)
        for fold_idx, (_, fold_test_idx) in enumerate(
                skf.split(np.arange(len(ar_summary)), ar_summary["_stratum"].values)):
            fold_assignments[fold_test_idx] = fold_idx
        ar_summary["_fold"] = fold_assignments

        ar_to_fold  = ar_summary.set_index("ar_number")["_fold"].to_dict()
        df          = df.copy()
        df["_fold"] = df["ar_number"].map(ar_to_fold)

        test_fold       = val_split % N_SPLITS
        pseudotest_fold = (val_split + 1) % N_SPLITS
        val_fold        = (val_split + 2) % N_SPLITS
        train_folds     = [f for f in range(N_SPLITS)
                           if f not in (test_fold, pseudotest_fold, val_fold)]

        df_test       = df[df["_fold"] == test_fold      ].drop(columns="_fold")
        df_pseudotest = df[df["_fold"] == pseudotest_fold].drop(columns="_fold")
        df_val        = df[df["_fold"] == val_fold       ].drop(columns="_fold")
        df_train      = df[df["_fold"].isin(train_folds) ].drop(columns="_fold")

        return df_test, df_pseudotest, df_train, df_val

    # ------------------------------------------------------------------ #
    # SIMPLE TEMPORAL SPLIT FOR SMALL DATASETS                            #
    # ------------------------------------------------------------------ #

    if len(df) < 365:
        print(f"WARNING: Dataset very small ({len(df)} samples). Using simple split.")
        n             = len(df)
        n_test        = max(1, int(n * 0.2))
        n_pseudo      = max(1, int(n * 0.1))
        n_val         = max(1, int((n - n_test - n_pseudo) * 0.2))

        df            = df.sort_values('sample_time').reset_index(drop=True)
        df_test       = df.iloc[-n_test:]
        df_pseudotest = df.iloc[-(n_test + n_pseudo):-n_test]
        df_remaining  = df.iloc[:-(n_test + n_pseudo)]
        df_val        = df_remaining.iloc[-n_val:]
        df_train      = df_remaining.iloc[:-n_val]

        return df_test, df_pseudotest, df_train, df_val

    # ------------------------------------------------------------------ #
    # TEST A / TEST B / DEFAULT                                           #
    # ------------------------------------------------------------------ #

    inds_test_a = (df['sample_time'].dt.month >= 11)
    inds_test_b = (df['sample_time'] >= datetime(2021, 1, 1)) & \
                  (df['sample_time'] <  datetime(2023, 1, 1))

    if test == 'test_a':
        inds_test = inds_test_a
    elif test == 'test_b':
        inds_test = inds_test_b
    else:
        inds_test = inds_test_a | inds_test_b

    df_test = df.loc[inds_test, :]
    df_full = df.loc[~inds_test, :]

    if test == 'test_a':
        inds_pseudotest = (
            ((df_full['sample_time'].dt.month == 10) & (df_full['sample_time'].dt.day > 26)) |
            ((df_full['sample_time'].dt.month == 1)  & (df_full['sample_time'].dt.day < 6))
        )
    elif test == 'test_b':
        inds_pseudotest = (df['sample_time'] >= datetime(2020, 12, 26))
    else:
        inds_pseudotest = (
            (df_full['sample_time'].dt.month == 10) |
            ((df_full['sample_time'].dt.month == 9) & (df_full['sample_time'].dt.day > 15))
        )

    df_pseudotest = df_full.loc[inds_pseudotest, :]
    df_train      = df_full.loc[~inds_pseudotest, :].reset_index(drop=True)
    n_val         = int(np.floor(len(df_train) / 5))
    df_val        = df_train.iloc[val_split * n_val:(val_split + 1) * n_val, :]
    df_train      = df_train.drop(df_val.index)

    return df_test, df_pseudotest, df_train, df_val


def split_data_overfit(df):
    """Split dataset — overfitting test mode (same data for all splits)."""
    print("WARNING: Overfitting test mode - using same data for train/val/test")
    df_small = df.copy()
    return df_small, df_small, df_small, df_small


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MagnetogramMultiDataSet(Dataset):
    """
    Dataset for multi-output Poisson flare-count prediction.

    Returns per sample:
        filename  (str)
        image     (Tensor 1×H×W)
        features  (Tensor n_features)   — zeros if no feature_cols
        label     (Tensor 3)            — raw integer counts [C, M, X]
    """

    TARGET_COLS = ['C_count', 'M_count', 'X_count']

    def __init__(self, df, target_cols: list, transform=transforms.ToTensor(),
                 feature_cols: list = [], file_col: str = 'filename',
                 maxval: float = 300):
        self.file_col     = file_col
        self.name_frame   = df[file_col].reset_index(drop=True)
        self.label_frame  = df[target_cols].values.astype(np.float32)
        self.transform    = transform
        self.features     = df[feature_cols].copy().reset_index(drop=True) if feature_cols else None
        self.feature_cols = feature_cols
        self.maxval       = maxval

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        filename = self.name_frame.iloc[idx]

        if self.file_col != 'filename':
            img = np.load('../solar-similarity-search/' + filename).astype(np.float32)
            img = img / self.maxval
        else:
            with h5py.File(filename, 'r') as f:
                img = np.array(f['magnetogram']).astype(np.float32)
            img = np.nan_to_num(img)
            img = np.clip(img, -self.maxval, self.maxval) / self.maxval

        img   = self.transform(img)
        label = torch.tensor(self.label_frame[idx], dtype=torch.float32)

        if self.feature_cols:
            features = torch.tensor(self.features.iloc[idx].to_numpy(), dtype=torch.float32)
        else:
            features = torch.zeros(0, dtype=torch.float32)

        return filename, img, features, label


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODULE
# ─────────────────────────────────────────────────────────────────────────────

class MagnetogramMultiDataModule(pl.LightningDataModule):
    """
    DataModule for multi-output Poisson count prediction: (C_count, M_count, X_count).

    Targets are raw integer counts — no transformation applied.
    The model's final head uses softplus to guarantee λ > 0.
    Loss: sum of per-class negative Poisson log-likelihoods.
    """

    def __init__(
        self,
        data_file: str,
        target_cols: list = None,
        feature_cols: list = None,
        val_split: int = 1,
        forecast_window: int = 24,
        dim: int = 256,
        batch: int = 32,
        augmentation: str = None,
        test: str = '',
        file_col: str = 'filename',
        maxval: float = 300,
        balance_ratio: int = None,
        regression: bool = True,
        flare_thresh: float = 1e-5,
        flux_thresh: float = 1.5e7,
        label: str = None,
    ):
        super().__init__()
        self.data_file    = data_file
        self.target_cols  = target_cols or ['C_count', 'M_count', 'X_count']
        self.feature_cols = feature_cols or []
        self.val_split    = val_split
        self.dim          = dim
        self.batch_size   = batch
        self.augmentation = augmentation
        self.test         = test
        self.file_col     = file_col
        self.maxval       = maxval

        base_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((dim, dim * 2),
                              transforms.InterpolationMode.BILINEAR, antialias=True),
        ])
        self.transform = base_tf

        if augmentation == 'conservative':
            self.training_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((dim, dim * 2),
                                  transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.RandomVerticalFlip(p=0.5),
                RandomPolaritySwitch(p=0.5),
            ])
        elif augmentation == 'intermediate':
            self.training_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((int(dim * 1.2), int(dim * 2 * 1.2)),
                                  transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.RandomVerticalFlip(p=0.5),
                RandomPolaritySwitch(p=0.5),
                transforms.RandomCrop((dim, dim * 2)),
            ])
        elif augmentation == 'full':
            self.training_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((dim, dim * 2),
                                  transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                RandomPolaritySwitch(p=0.5),
            ])
        else:
            self.training_transform = self.transform

    def prepare_data(self):
        self.df = pd.read_csv(self.data_file)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'], format='mixed')

        for col in self.target_cols:
            if col not in self.df.columns:
                raise ValueError(
                    f"Target column '{col}' not found. "
                    f"Available: {list(self.df.columns)}")

        self.df = self.df.dropna(subset=self.target_cols).reset_index(drop=True)

        # Readable summary: positives/negatives per class, mean λ, max count
        _print_flare_summary("Full dataset", self.df, self.target_cols)

    def setup(self, stage: str):
        df_test, df_pseudotest, df_train, df_val = split_data(
            self.df, self.val_split, self.test)

        if self.feature_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(df_train[self.feature_cols])
            for split in [df_train, df_val, df_pseudotest, df_test]:
                split[self.feature_cols] = self.scaler.transform(
                    split[self.feature_cols])

        def make_ds(df, transform):
            return MagnetogramMultiDataSet(
                df, self.target_cols, transform,
                self.feature_cols, self.file_col, self.maxval)

        self.train_set      = make_ds(df_train,                      self.training_transform)
        self.val_set        = make_ds(df_val,                        self.transform)
        self.trainval_set   = make_ds(pd.concat([df_train, df_val]), self.transform)
        self.pseudotest_set = make_ds(df_pseudotest,                 self.transform)
        self.test_set       = make_ds(df_test,                       self.transform)

        print(f'\nSplit sizes — Train: {len(self.train_set)}  '
              f'Val: {len(self.val_set)}  '
              f'PseudoTest: {len(self.pseudotest_set)}  '
              f'Test: {len(self.test_set)}')

        # Readable summary for each split
        for split_name, sdf in [('Train',      df_train),
                                 ('Val',        df_val),
                                 ('PseudoTest', df_pseudotest),
                                 ('Test',       df_test)]:
            _print_flare_summary(split_name, sdf, self.target_cols)

    def train_dataloader(self):
        return DataLoader(self.train_set,  batch_size=self.batch_size,
                          num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,    batch_size=self.batch_size,
                          num_workers=4, drop_last=True)

    def trainval_dataloader(self, shuffle=False):
        return DataLoader(self.trainval_set, batch_size=self.batch_size,
                          num_workers=4, shuffle=shuffle, drop_last=True)

    def pseudotest_dataloader(self):
        return DataLoader(self.pseudotest_set, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set,   batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test_set,   batch_size=self.batch_size, num_workers=4)