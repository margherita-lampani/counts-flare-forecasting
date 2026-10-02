"""
Run multi-seed k-fold cross-validation experiments for solar flare forecasting.

This script performs repeated 5-fold cross-validation across multiple random
seeds for both flare-count regression (Poisson) and flare-occurrence
classification (Binary) tasks. It manages data loading, model training,
checkpoint selection, prediction calibration, performance evaluation, and
result aggregation.

Main features:
    - Multi-seed reproducible experiments
    - 5-fold cross-validation with fixed test partition
    - Poisson regression and binary-classification modes
    - Automatic checkpointing and early stopping
    - Post-training calibration of decision thresholds
    - Per-split evaluation on train, validation, pseudotest, and test sets
    - Optional experiment tracking with Comet ML

Metrics:
    Classification:
        TSS, HSS2, AUC

    Regression (Poisson mode):
        MAE, MSE, RMSE, Bias, Pearson correlation,
        Poisson deviance, and outlier statistics

Outputs:
    results/metrics_<tag>_seed_<seed>.csv
        Fold-level performance metrics.

    results/predictions_<tag>_seed_<seed>.csv
        Sample-level predictions and ground-truth labels.

The generated files constitute the primary experimental outputs and are
subsequently aggregated by the analysis pipeline to obtain final performance
statistics across folds and random seeds.
"""

import sys, os, argparse, copy

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

try:
    from comet_ml import ExistingExperiment
    from pytorch_lightning.loggers import CometLogger
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.benchmark    = False
torch.backends.cudnn.deterministic = True

from flare_forecast.data.dataset         import MagnetogramMultiDataModule
from flare_forecast.models.poisson        import FlexConvNetMulti,  LitPoissonMulti
from flare_forecast.models.binary import FlexConvNetBinary, LitBinary
from flare_forecast.training.callbacks    import TSSCallback, confusion_skills
from flare_forecast.training.train import (build_model, calibrate_poisson,
                                       calibrate_binary, FIXED_ARCH, CLASS_NAMES)

N_FOLDS = 5


# ─────────────────────────────────────────────────────────────────────────────
# TAG BUILDER  (unique identifier for this experimental configuration)
# ─────────────────────────────────────────────────────────────────────────────

def build_tag(config: dict, mode: str) -> str:
    """
    Build a short string that uniquely identifies the monitor metric and the
    set of scalar features used.  Used as part of every output filename so
    that results from different configurations never collide.

    Examples
    --------
    "val_TSS_C_feat-tot_us_flux+tot_flux+datamin+datamax"
    "val_TSS_C_feat-NONE"
    "val_loss_feat-NONE"  (binary with no features)
    """
    tc = config['training']
    if mode == 'poisson':
        monitor = tc.get('monitor_metric', 'val_TSS_C')
    else:
        monitor = tc.get('monitor_metric_binary', 'val_loss')

    feats = config['data'].get('feature_cols') or []
    feat_str = '+'.join(feats) if feats else 'NONE'

    return f"{monitor}_feat-{feat_str}"


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED POISSON REGRESSION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray,
                     eps: float = 1e-8) -> float:
    """
    Poisson deviance = 2 * sum( y*log(y/ŷ) - (y - ŷ) ).
    Returned as the *mean* deviance per sample (easier to compare across
    differently-sized splits).

    Notes
    -----
    - For samples where y=0 the term y*log(y/ŷ) = 0 by convention.
    - ŷ is clipped to eps to avoid log(0).
    """
    lam = np.clip(y_pred, eps, None)
    # term: y*log(y/lam) — set to 0 when y==0
    ratio_term = np.where(y_true > 0, y_true * np.log(y_true / lam), 0.0)
    dev = 2.0 * (ratio_term - (y_true - lam))
    return float(dev.mean())


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       eps: float = 1e-8) -> dict:
    """
    Compute a full suite of regression metrics for one class.

    Returns
    -------
    dict with keys:
        MAE, MSE, RMSE, Bias, PearsonR, PearsonR_p,
        PoissonDeviance, FracOutliers2x
    """
    residuals = y_pred - y_true                          # signed error
    abs_err   = np.abs(residuals)

    mae  = float(abs_err.mean())
    mse  = float((residuals ** 2).mean())
    rmse = float(np.sqrt(mse))
    bias = float(residuals.mean())                       # mean signed error

    # Pearson r between λ and true count (over ALL samples, including zeros)
    if y_true.std() > 0 and y_pred.std() > 0:
        r, p_val = scipy_stats.pearsonr(y_true, y_pred)
        pearson_r  = float(r)
        pearson_p  = float(p_val)
    else:
        pearson_r  = float('nan')
        pearson_p  = float('nan')

    dev = poisson_deviance(y_true, y_pred, eps=eps)

    # Fraction of samples where |error| > 2 * y_true  (only where y_true > 0)
    pos_mask = y_true > 0
    if pos_mask.sum() > 0:
        frac_out = float((abs_err[pos_mask] > 2.0 * y_true[pos_mask]).mean())
    else:
        frac_out = float('nan')

    return dict(
        MAE            = mae,
        MSE            = mse,
        RMSE           = rmse,
        Bias           = bias,
        PearsonR       = pearson_r,
        PearsonR_p     = pearson_p,
        PoissonDeviance= dev,
        FracOutliers2x = frac_out,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION METRICS  (shared by both modes)
# ─────────────────────────────────────────────────────────────────────────────

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            threshold: float) -> dict:
    """
    Binarise predictions at `threshold` and compute confusion-matrix skills
    plus AUC (if both classes present).
    """
    obs      = (y_true >= 0.5).astype(int)
    pred_bin = (y_pred >= threshold).astype(int)
    s        = confusion_skills(obs, pred_bin)

    auc = float('nan')
    if obs.sum() > 0 and (1 - obs).sum() > 0:
        auc = float(roc_auc_score(obs, y_pred))

    return {**s, 'AUC': auc}


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT PER-SAMPLE PREDICTIONS FROM ONE DATALOADER
# ─────────────────────────────────────────────────────────────────────────────

def collect_predictions(trainer, lit_model, loader,
                        split: str, seed: int, fold: int, mode: str) -> list:
    """
    Run trainer.predict on `loader` and return a list of flat dicts,
    one per sample.  Each dict contains filename, seed, fold, mode, split,
    lam/prob per class (yp), and ground-truth count (yt).
    """
    raw     = trainer.predict(model=lit_model, dataloaders=loader)
    records = []
    for batch in raw:
        fnames = batch[0]
        yt = batch[1].cpu().float().numpy()
        yp = batch[2].cpu().float().numpy()
        for i, fname in enumerate(fnames):
            rec = dict(filename=fname, seed=seed, fold=fold,
                       mode=mode, split=split)
            for j, cls in enumerate(CLASS_NAMES):
                rec[f'yp_{cls}'] = float(yp[i, j])
                rec[f'yt_{cls}'] = float(yt[i, j])
            records.append(rec)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE FOLD
# ─────────────────────────────────────────────────────────────────────────────

def run_fold(config: dict, mode: str, fold: int, seed: int,
             comet_logger=None, experiment=None) -> tuple[dict, list]:
    """
    Train and evaluate one fold.

    Returns
    -------
    fold_metrics : dict   — scalar metrics (one row for the metrics CSV)
    pred_records : list   — per-sample dicts  (rows for the predictions CSV)
    """
    print(f"\n{'='*60}")
    print(f"  SEED {seed}  |  FOLD {fold}  |  MODE: {mode.upper()}")
    print(f"{'='*60}\n")

    pl.seed_everything(seed, workers=True)

    cfg = copy.deepcopy(config)
    cfg['data']['val_split'] = fold

    data = MagnetogramMultiDataModule(
        data_file       = cfg['data']['data_file'],
        target_cols     = cfg['data'].get('target_cols', ['C_count','M_count','X_count']),
        feature_cols    = cfg['data'].get('feature_cols') or [],
        val_split       = fold,
        forecast_window = cfg['data']['forecast_window'],
        dim             = cfg['data']['dim'],
        batch           = cfg['training']['batch_size'],
        augmentation    = cfg['data']['augmentation'],
        test            = cfg['data']['test'],
        file_col        = cfg['data']['file_col'],
        maxval          = cfg['data']['maxval'],
    )
    data.prepare_data()
    data.setup('fit')

    backbone, lit_model = build_model(cfg, mode, data)

    tc = cfg['training']
    if mode == 'poisson':
        monitor = tc.get('monitor_metric', 'val_TSS_C')
    else:
        monitor = tc.get('monitor_metric_binary', 'val_loss')
    mon_mode = 'min' if any(s in monitor for s in ('loss', 'mae')) else 'max'

    ckpt_dir = Path('checkpoints') / mode / f'seed{seed}' / f'fold{fold}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        accelerator         = tc['device'],
        devices             = tc.get('devices', 1),
        max_epochs          = tc['epochs'],
        callbacks           = [
            ModelSummary(max_depth=1),
            ModelCheckpoint(
                dirpath   = str(ckpt_dir),
                monitor   = monitor, mode=mon_mode,
                save_top_k= 1, save_last=True, save_weights_only=True,
                filename  = f'{mode}-seed{seed}-fold{fold}'
                            f'-{{epoch:02d}}-{{{monitor}:.4f}}'),
            EarlyStopping(
                monitor=monitor, min_delta=0.0005,
                patience=tc.get('patience', 20),
                mode=mon_mode, strict=False, check_finite=False),
            TSSCallback(
                primary_key=tc.get('tss_primary_key', 'M'),
                mode=mode),
        ],
        logger              = comet_logger,
        enable_progress_bar = True,
        precision           = 32,
        deterministic       = True,
    )

    trainer.fit(model=lit_model, datamodule=data)

    # ── Load best checkpoint ──────────────────────────────────────────────
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\nBest checkpoint: {best_ckpt}")
    if best_ckpt and Path(best_ckpt).exists():
        cls_lit = LitPoissonMulti if mode == 'poisson' else LitBinary
        try:
            lit_model = cls_lit.load_from_checkpoint(best_ckpt, model=backbone)
        except Exception as e:
            print(f"  Strict load failed ({e}), retrying with strict=False ...")
            lit_model = cls_lit.load_from_checkpoint(
                best_ckpt, model=backbone, strict=False)

    # ── Calibration on validation set ────────────────────────────────────
    print(f"\nCalibrating (seed={seed}, fold={fold}, mode={mode})...")
    val_preds = trainer.predict(model=lit_model, dataloaders=data.val_dataloader())
    calib = (calibrate_poisson(val_preds,
                               tau_min=cfg['testing'].get('tau_min', 0.01),
                               tau_max=cfg['testing'].get('tau_max', 2.0),
                               n_tau  =cfg['testing'].get('n_tau', 500))
             if mode == 'poisson'
             else calibrate_binary(val_preds))

    thr = {cls: calib.get(cls, {}).get('tau_star', 0.5) for cls in CLASS_NAMES}

    # ── Evaluate all splits ───────────────────────────────────────────────
    fold_metrics  = {'seed': seed, 'fold': fold, 'mode': mode,
                     'monitor': monitor,
                     'tau_C': thr['C'], 'tau_M': thr['M'], 'tau_X': thr['X']}
    all_pred_records: list[dict] = []

    split_loaders = [
        ('train',      data.train_dataloader()),
        ('val',        data.val_dataloader()),
        ('pseudotest', data.pseudotest_dataloader()),
        ('test',       data.test_dataloader()),
    ]

    for split, loader in split_loaders:
        preds  = trainer.predict(model=lit_model, dataloaders=loader)
        yt_all = np.concatenate([b[1].cpu().float().numpy() for b in preds])
        yp_all = np.concatenate([b[2].cpu().float().numpy() for b in preds])

        # Collect per-sample rows
        recs = collect_predictions(trainer, lit_model, loader,
                                   split, seed, fold, mode)
        all_pred_records.extend(recs)

        for j, cls in enumerate(CLASS_NAMES):
            yt, yp = yt_all[:, j], yp_all[:, j]

            # ── Classification metrics (both modes) ───────────────────────
            cm = classification_metrics(yt, yp, thr[cls])
            for k, v in cm.items():
                fold_metrics[f'{split}_{k}_{cls}'] = v

            # ── Regression metrics (Poisson mode only) ────────────────────
            if mode == 'poisson':
                rm = regression_metrics(yt, yp)
                for k, v in rm.items():
                    fold_metrics[f'{split}_{k}_{cls}'] = v

        # ── Concise per-split print ───────────────────────────────────────
        tss_line = '  '.join(
            f"TSS_{c}={fold_metrics.get(f'{split}_TSS_{c}', float('nan')):.4f}"
            for c in CLASS_NAMES)
        print(f"\n  [{split} | seed {seed} | fold {fold}]  {tss_line}")
        if mode == 'poisson':
            mae_line = '  '.join(
                f"MAE_{c}={fold_metrics.get(f'{split}_MAE_{c}', float('nan')):.4f}"
                for c in CLASS_NAMES)
            dev_line = '  '.join(
                f"Dev_{c}={fold_metrics.get(f'{split}_PoissonDeviance_{c}', float('nan')):.4f}"
                for c in CLASS_NAMES)
            print(f"              {mae_line}")
            print(f"              {dev_line}")

    if experiment:
        for k, v in fold_metrics.items():
            if k not in ('seed', 'fold', 'mode', 'monitor'):
                experiment.log_metric(f'seed{seed}_fold{fold}_{k}', v)

    return fold_metrics, all_pred_records


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY (printed at end of each seed run)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, mode: str):
    sub    = df[df['mode'] == mode]
    n_runs = len(sub)
    print(f"\n{'='*65}")
    print(f"  SUMMARY — {mode.upper()}  ({n_runs} fold-runs for this seed)")
    print(f"{'='*65}")
    print(f"  {'Metric':<32}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*32}  {'-'*8}  {'-'*8}")

    clf_metrics = ['TSS', 'HSS2', 'AUC']
    reg_metrics = ['MAE', 'MSE', 'RMSE', 'Bias',
                   'PearsonR', 'PoissonDeviance', 'FracOutliers2x']
    all_metrics = clf_metrics + (reg_metrics if mode == 'poisson' else [])

    for cls in CLASS_NAMES:
        for metric in all_metrics:
            col = f'test_{metric}_{cls}'
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals):
                    print(f"  {col:<32}  {vals.mean():>8.4f}  {vals.std():>8.4f}")
    print(f"{'='*65}\n")



def run_one_seed(config: dict, seed: int, modes: list, folds: list,
                 out_dir, shared_tag: str):
    """
    Run all modes x folds for a single seed, save per-seed CSVs,
    and print a quick summary.

    If the metrics CSV for this seed already exists, the seed is skipped
    (resume-safe: re-launching after a crash restarts from the first
    incomplete seed).
    """
    metrics_csv = out_dir / f'metrics_{shared_tag}_seed_{seed}.csv'
    preds_csv   = out_dir / f'predictions_{shared_tag}_seed_{seed}.csv'

    if metrics_csv.exists():
        print(f"\n  [SKIP] Seed {seed} already complete ({metrics_csv.name})")
        existing_metrics = pd.read_csv(metrics_csv).to_dict('records')
        existing_preds   = (pd.read_csv(preds_csv).to_dict('records')
                            if preds_csv.exists() else [])
        return existing_metrics, existing_preds

    print(f"\n{'#'*65}")
    print(f"  SEED {seed}  |  {len(modes)} mode(s) x {len(folds)} fold(s)"
          f" = {len(modes) * len(folds)} runs")
    print(f"{'#'*65}\n")

    all_metrics = []
    all_preds   = []

    for mode in modes:
        comet_logger, experiment = None, None
        if COMET_AVAILABLE and config.get('comet', {}).get('enabled', False):
            cc       = config['comet']
            exp_name = (f"{cc.get('experiment_name', '')}_{mode}"
                        f"_seed{seed}").lstrip('_')
            comet_logger = CometLogger(
                project_name    = cc['project_name'],
                workspace       = cc.get('workspace'),
                experiment_name = exp_name,
            )
            experiment = comet_logger.experiment
            experiment.log_parameters(config)
            experiment.log_parameters(FIXED_ARCH)
            experiment.log_other('mode',    mode)
            experiment.log_other('seed',    seed)
            experiment.log_other('n_folds', len(folds))
            experiment.log_other('tag',     shared_tag)
            print(f"Comet: {experiment.url}")

        for fold in folds:
            fold_metrics, pred_records = run_fold(
                config, mode, fold, seed,
                comet_logger=comet_logger,
                experiment=experiment)
            all_metrics.append(fold_metrics)
            all_preds.extend(pred_records)

        if experiment:
            experiment.end()

    # Save per-seed CSVs
    metrics_csv = out_dir / f'metrics_{shared_tag}_seed_{seed}.csv'
    preds_csv   = out_dir / f'predictions_{shared_tag}_seed_{seed}.csv'
    pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)
    pd.DataFrame(all_preds).to_csv(preds_csv,     index=False)
    print(f"\n  Seed {seed} - metrics     -> {metrics_csv}")
    print(f"  Seed {seed} - predictions -> {preds_csv}")

    df_metrics = pd.DataFrame(all_metrics)
    for mode in modes:
        print_summary(df_metrics, mode)

    return all_metrics, all_preds


def main():
    parser = argparse.ArgumentParser(
        description='K-fold CV across multiple seeds (sequential, one GPU).')
    parser.add_argument('--config', default='configs/default.yml')
    parser.add_argument('--seeds',  nargs='+', type=int, default=list(range(10)),
                        help='Seeds to run sequentially (default: 0-9)')
    parser.add_argument('--modes',  nargs='+', default=['poisson', 'binary'],
                        choices=['poisson', 'binary'])
    parser.add_argument('--folds',  nargs='+', type=int, default=list(range(N_FOLDS)))
    parser.add_argument('--merge',  action='store_true',
                        help='Auto-run merge_results when all seeds finish')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config['data']['test'] = config['data']['test'].lower()

    out_dir = Path('results')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build shared tag once
    tc       = config['training']
    monitor  = tc.get('monitor_metric') or tc.get('monitor_metric_binary', 'val_loss')
    feats    = config['data'].get('feature_cols') or []
    feat_str = '+'.join(feats) if feats else 'NONE'
    shared_tag = f"{monitor}_feat-{feat_str}"

    total_runs = len(args.seeds) * len(args.modes) * len(args.folds)
    print(f"\nConfiguration tag : {shared_tag}")
    print(f"Seeds             : {args.seeds}")
    print(f"Modes             : {args.modes}")
    print(f"Folds             : {args.folds}")
    print(f"Total runs        : {total_runs}  (sequential, one GPU)\n")

    all_metrics_global = []
    all_preds_global   = []

    for i, seed in enumerate(args.seeds):
        print(f"\n[{i+1}/{len(args.seeds)}] Starting seed {seed}...")
        seed_metrics, seed_preds = run_one_seed(
            config, seed, args.modes, args.folds, out_dir, shared_tag)
        all_metrics_global.extend(seed_metrics)
        all_preds_global.extend(seed_preds)

    # Final summary across all seeds
    print(f"\n{'#'*65}")
    print(f"  ALL SEEDS COMPLETE")
    print(f"{'#'*65}")
    df_all = pd.DataFrame(all_metrics_global)
    for mode in args.modes:
        print_summary(df_all, mode)

    if args.merge:
        import sys as _sys, importlib
        _sys.argv = ['merge_results', '--tag', shared_tag]
        importlib.import_module('flare_forecast.analysis.merge_results').main()


if __name__ == '__main__':
    main()