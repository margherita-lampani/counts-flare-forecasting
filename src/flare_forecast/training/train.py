"""
Unified training and evaluation pipeline for solar flare forecasting.

This module orchestrates the complete model-development workflow, including
data loading, model construction, training, checkpoint selection, threshold
calibration, and final performance evaluation. Both Poisson count forecasting
and binary flare-occurrence prediction are supported within a common framework.

Main features:
    - End-to-end training pipeline based on PyTorch Lightning
    - Support for Poisson regression and binary classification modes
    - Fixed architecture derived from prior hyperparameter optimization
    - Automatic checkpointing and early stopping
    - Validation-based threshold calibration
    - Multi-class forecasting for C-, M-, and X-class flares
    - Experiment tracking and metadata logging through Comet ML

Forecasting modes:
    Poisson mode:
        Predicts the expected number of future flares
        (λ_C, λ_M, λ_X) using Poisson regression.

    Binary mode:
        Predicts the probability of future flare occurrence
        P(C≥1), P(M≥1), and P(X≥1).

Calibration:
    Model outputs are calibrated using the validation set to determine
    class-specific decision thresholds that maximize the True Skill
    Statistic (TSS). The calibrated thresholds are subsequently used
    during performance evaluation on all dataset splits.

Evaluation:
    Metrics are computed for train, validation, pseudotest, and test sets.

    Classification:
        TSS, HSS2, AUC, and confusion-matrix statistics.

    Regression (Poisson mode):
        MAE and coefficient of determination (R²).

Outputs:
    - Trained model checkpoints
    - Calibrated decision thresholds
    - Experiment logs and performance metrics stored in Comet ML

This module serves as the central training entry point for all flare
forecasting experiments.
"""

import sys, os, argparse, math, json

import yaml
import numpy as np
from pathlib import Path

try:
    from comet_ml import ExistingExperiment
    from pytorch_lightning.loggers import CometLogger
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping

from flare_forecast.data.dataset         import MagnetogramMultiDataModule
from flare_forecast.models.poisson        import FlexConvNetMulti,  LitPoissonMulti
from flare_forecast.models.binary import FlexConvNetBinary, LitBinary
from flare_forecast.training.callbacks    import TSSCallback, confusion_skills

CLASS_NAMES = ['C', 'M', 'X']

# ── Fixed architecture (same as optuna_search.py) ─────────────────────────────
FIXED_ARCH = dict(
    n_blocks      = 5,
    base_channels = 16,
    fc_hidden     = 256,
    head_hidden   = 64,
    kernel_size   = 5,
    dropout_ratio = 0.45,
    pooling_type  = 'gap',
    use_skip      = True,
)


# ─────────────────────────────────────────────────────────────────────────────
# SKILL SCORES
# ─────────────────────────────────────────────────────────────────────────────

def _skill(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> dict:
    obs  = (y_true >= 0.5).astype(int)
    pred = (y_pred >= tau).astype(int)
    return confusion_skills(obs, pred)


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION — Poisson
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_poisson(val_preds, tau_min=0.01, tau_max=2.0, n_tau=500,
                      experiment=None) -> dict:
    yt_all = np.concatenate([b[1].cpu().float().numpy() for b in val_preds])
    yp_all = np.concatenate([b[2].cpu().float().numpy() for b in val_preds])
    results = {}

    for cls, idx in [('C', 0), ('M', 1), ('X', 2)]:
        yt, yp = yt_all[:, idx], yp_all[:, idx]
        taus   = np.linspace(tau_min,
                             min(tau_max, float(np.percentile(yp, 99)) + 0.1),
                             n_tau)
        tss_v    = np.array([_skill(yt, yp, t)['TSS'] for t in taus])
        tau_star = float(taus[np.argmax(tss_v)])
        s        = _skill(yt, yp, tau_star)
        s_def    = _skill(yt, yp, 0.5)

        print(f"  [calib poisson | {cls}]  τ*={tau_star:.4f}  "
              f"TSS={s['TSS']:.4f}  HSS2={s['HSS2']:.4f}  "
              f"TP={s['TP']}  FP={s['FP']}  FN={s['FN']}  TN={s['TN']}  "
              f"(τ=0.5 → TSS={s_def['TSS']:.4f})")

        results[cls] = {**s, 'tau_star': tau_star,
                        'TSS_at_0.5': s_def['TSS'], 'HSS2_at_0.5': s_def['HSS2'],
                        'n_pos': int((yt >= 0.5).sum()),
                        'n_neg': int((yt < 0.5).sum())}
        if experiment:
            for k, v in results[cls].items():
                experiment.log_metric(f'calib_poisson_{cls}_{k}', v)

    if experiment:
        experiment.log_asset_data(json.dumps(results, indent=2),
                                  name='optimal_thresholds_poisson.json')
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION — Binary
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_binary(val_preds, experiment=None) -> dict:
    from sklearn.metrics import roc_curve

    yt_all = np.concatenate([b[1].cpu().float().numpy() for b in val_preds])
    yp_all = np.concatenate([b[2].cpu().float().numpy() for b in val_preds])
    results = {}

    for cls, idx in [('C', 0), ('M', 1), ('X', 2)]:
        yt, yp = yt_all[:, idx], yp_all[:, idx]
        t_bin  = (yt >= 0.5).astype(int)

        if t_bin.sum() == 0 or (1 - t_bin).sum() == 0:
            print(f"  [calib binary  | {cls}]  skipped (single class)")
            results[cls] = {'tau_star': 0.5}
            continue

        fpr, tpr, thresholds = roc_curve(t_bin, yp)
        tau_star = float(thresholds[int(np.argmax(tpr - fpr))])
        s        = _skill(yt, yp, tau_star)
        s_def    = _skill(yt, yp, 0.5)

        print(f"  [calib binary  | {cls}]  τ*={tau_star:.4f}  "
              f"TSS={s['TSS']:.4f}  HSS2={s['HSS2']:.4f}  "
              f"TP={s['TP']}  FP={s['FP']}  FN={s['FN']}  TN={s['TN']}  "
              f"(τ=0.5 → TSS={s_def['TSS']:.4f})")

        results[cls] = {**s, 'tau_star': tau_star,
                        'TSS_at_0.5': s_def['TSS'], 'HSS2_at_0.5': s_def['HSS2'],
                        'n_pos': int(t_bin.sum()),
                        'n_neg': int((1 - t_bin).sum())}
        if experiment:
            for k, v in results[cls].items():
                experiment.log_metric(f'calib_binary_{cls}_{k}', v)

    if experiment:
        experiment.log_asset_data(json.dumps(results, indent=2),
                                  name='optimal_thresholds_binary.json')
    return results


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE + LOG TO COMET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_log(preds, split_name: str, mode: str,
                     calibrated_thresholds: dict, experiment=None):
    from sklearn.metrics import roc_auc_score

    thr = {cls: calibrated_thresholds.get(cls, {}).get('tau_star', 0.5)
           for cls in CLASS_NAMES}

    yt_all = np.concatenate([b[1].cpu().float().numpy() for b in preds])
    yp_all = np.concatenate([b[2].cpu().float().numpy() for b in preds])

    print(f"\n  [{split_name} | {mode}]  n={len(yt_all)}  "
          f"τ_C={thr['C']:.4f}  τ_M={thr['M']:.4f}  τ_X={thr['X']:.4f}")

    metrics = {}
    for j, cls in enumerate(CLASS_NAMES):
        yt, yp = yt_all[:, j], yp_all[:, j]
        s = _skill(yt, yp, thr[cls])
        for k, v in s.items():
            metrics[f'{cls}_{k}'] = v

        if (yt >= 0.5).sum() > 0 and (yt < 0.5).sum() > 0:
            metrics[f'{cls}_AUC'] = float(
                roc_auc_score((yt >= 0.5).astype(int), yp))

        if mode == 'poisson':
            metrics[f'{cls}_MAE'] = float(np.abs(yt - yp).mean())
            metrics[f'{cls}_R2']  = float(
                1 - np.sum((yt - yp)**2) / (np.sum((yt - np.mean(yt))**2) + 1e-9))

    for cls in CLASS_NAMES:
        line = (f"    {cls}:  TSS={metrics.get(f'{cls}_TSS', float('nan')):.4f}  "
                f"HSS2={metrics.get(f'{cls}_HSS2', float('nan')):.4f}  "
                f"AUC={metrics.get(f'{cls}_AUC', float('nan')):.4f}")
        if mode == 'poisson':
            line += (f"  MAE={metrics.get(f'{cls}_MAE', float('nan')):.4f}  "
                     f"R²={metrics.get(f'{cls}_R2', float('nan')):.4f}")
        print(line)

    if experiment:
        for k, v in metrics.items():
            experiment.log_metric(f'{split_name}_{k}', v)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: dict, mode: str, data: MagnetogramMultiDataModule):
    """Build model using FIXED_ARCH — config model section ignored for arch."""
    labels = data.train_set.label_frame
    tc     = config['training']

    # Architecture always from FIXED_ARCH
    arch = dict(
        dim          = config['data']['dim'],
        length       = len(config['data']['channels']),
        #len_features = len(config['data']['feature_cols']),
        len_features = len(config['data'].get('feature_cols') or []),
        n_blocks      = FIXED_ARCH['n_blocks'],
        base_channels = FIXED_ARCH['base_channels'],
        fc_hidden     = FIXED_ARCH['fc_hidden'],
        head_hidden   = FIXED_ARCH['head_hidden'],
        pooling_type  = FIXED_ARCH['pooling_type'],
        kernel_size   = FIXED_ARCH['kernel_size'],
        dropoutRatio  = FIXED_ARCH['dropout_ratio'],
        use_skip      = FIXED_ARCH['use_skip'],
    )

    if mode == 'poisson':
        mean_C, mean_M, mean_X = (float(labels[:, i].mean()) for i in range(3))
        backbone  = FlexConvNetMulti(**arch,
                                     mean_C=mean_C, mean_M=mean_M, mean_X=mean_X)
        lit_model = LitPoissonMulti(
            model=backbone, lr=tc['lr'], wd=tc['wd'], epochs=tc['epochs'],
            scheduler_type=tc.get('scheduler', 'cosine_warmup'),
            w_C=float(tc.get('w_C', 1.0)),
            w_M=float(tc.get('w_M', 50.0)),
            w_X=float(tc.get('w_X', 200.0)),
            alpha_M=float(tc.get('alpha_M', 2.0)),
            alpha_X=float(tc.get('alpha_X', 4.0)),
        )

    elif mode == 'binary':
        pos_C = float((labels[:, 0] >= 1).mean())
        pos_M = float((labels[:, 1] >= 1).mean())
        pos_X = float((labels[:, 2] >= 1).mean())
        backbone  = FlexConvNetBinary(**arch,
                                      pos_rate_C=pos_C,
                                      pos_rate_M=pos_M,
                                      pos_rate_X=pos_X)
        lit_model = LitBinary(
            model=backbone, lr=tc['lr'], wd=tc['wd'], epochs=tc['epochs'],
            scheduler_type=tc.get('scheduler', 'cosine_warmup'),
            w_C=float(tc.get('w_C_binary', 1.0)),
            w_M=float(tc.get('w_M_binary', 30.0)),
            w_X=float(tc.get('w_X_binary', 300.0)),
        )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'poisson' or 'binary'.")

    return backbone, lit_model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yml')
    parser.add_argument('--mode',   default=None, choices=['poisson', 'binary'])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config['data']['test'] = config['data']['test'].lower()

    mode = args.mode or config.get('mode', 'poisson')
    print(f"\n{'='*60}\n  MODE: {mode.upper()}")
    print(f"  Architecture: {FIXED_ARCH}")
    print(f"{'='*60}\n")

    # ── Comet ─────────────────────────────────────────────────────────────
    comet_logger, experiment = None, None
    if COMET_AVAILABLE and config.get('comet', {}).get('enabled', False):
        cc = config['comet']
        exp_name = f"{cc.get('experiment_name', '')}_{mode}".lstrip('_')
        comet_logger = CometLogger(
            project_name=cc['project_name'],
            workspace=cc.get('workspace'), experiment_name=exp_name,
        )
        experiment = comet_logger.experiment
        experiment.log_parameters(config)
        experiment.log_parameters(FIXED_ARCH)
        experiment.log_other('mode', mode)
        if cc.get('log_code', True):
            experiment.log_code(folder="src")
        print(f"Comet: {experiment.url}")

    pl.seed_everything(42, workers=True)

    # ── Data ──────────────────────────────────────────────────────────────
    data = MagnetogramMultiDataModule(
        data_file       = config['data']['data_file'],
        target_cols     = config['data'].get('target_cols',
                                             ['C_count', 'M_count', 'X_count']),
        feature_cols    = config['data']['feature_cols'],
        val_split       = config['data']['val_split'],
        forecast_window = config['data']['forecast_window'],
        dim             = config['data']['dim'],
        batch           = config['training']['batch_size'],
        augmentation    = config['data']['augmentation'],
        test            = config['data']['test'],
        file_col        = config['data']['file_col'],
        maxval          = config['data']['maxval'],
    )
    data.prepare_data()
    data.setup('fit')

    backbone, lit_model = build_model(config, mode, data)

    # ── Monitor ───────────────────────────────────────────────────────────
    tc = config['training']
    if mode == 'poisson':
        monitor = tc.get('monitor_metric', 'val_mae_X')
    else:
        monitor = tc.get('monitor_metric_binary', 'val_loss')
    mon_mode = 'min' if any(s in monitor for s in ('loss', 'mae')) else 'max'

    print(f"  Monitor: {monitor}  (mode={mon_mode})")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator         = tc['device'],
        devices             = tc.get('devices', 1),
        max_epochs          = tc['epochs'],
        callbacks           = [
            ModelSummary(max_depth=2),
            ModelCheckpoint(
                dirpath=f'checkpoints/{mode}',
                monitor=monitor, mode=mon_mode, save_top_k=1,
                save_last=True, save_weights_only=True,
                filename=f'{mode}-{{epoch:02d}}-{{{monitor}:.4f}}'),
            EarlyStopping(
                monitor=monitor, min_delta=0.0005,
                patience=tc.get('patience', 30),
                mode=mon_mode, strict=False, check_finite=False),
            TSSCallback(
                primary_key=tc.get('tss_primary_key', 'M'),
                mode=mode),
        ],
        logger              = comet_logger,
        enable_progress_bar = True,
        precision           = 32,
    )

    trainer.fit(model=lit_model, datamodule=data)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\nBest checkpoint: {best_ckpt}")
    if experiment:
        experiment.log_other('best_checkpoint', best_ckpt)
    if best_ckpt and Path(best_ckpt).exists():
        cls = LitPoissonMulti if mode == 'poisson' else LitBinary
        lit_model = cls.load_from_checkpoint(best_ckpt, model=backbone)

    # ── Calibration ───────────────────────────────────────────────────────
    print(f"\nCalibrating thresholds on val ({mode})...")
    val_preds = trainer.predict(model=lit_model, dataloaders=data.val_dataloader())
    calib = (calibrate_poisson(val_preds,
                               tau_min=config['testing'].get('tau_min', 0.01),
                               tau_max=config['testing'].get('tau_max', 2.0),
                               n_tau  =config['testing'].get('n_tau', 500),
                               experiment=experiment)
             if mode == 'poisson'
             else calibrate_binary(val_preds, experiment=experiment))

    # ── Evaluation ────────────────────────────────────────────────────────
    print(f"\nEvaluation ({mode}):")
    test_exp = None
    if experiment and COMET_AVAILABLE:
        test_exp = ExistingExperiment(
            previous_experiment=experiment.id)

    for split, loader in [
        ('train',      data.train_dataloader()),
        ('val',        data.val_dataloader()),
        ('pseudotest', data.pseudotest_dataloader()),
        ('test',       data.test_dataloader()),
    ]:
        preds = trainer.predict(model=lit_model, dataloaders=loader)
        evaluate_and_log(preds, split, mode, calib, experiment=test_exp)

    if test_exp:   test_exp.end()
    if experiment: experiment.end()


if __name__ == '__main__':
    main()