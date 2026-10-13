"""
optuna_search.py
────────────────
Targeted Optuna search — architecture fixed, only training hyperparameters
searched.

Objective
─────────
Minimise:
    objective = val_mae_MX + pseudotest_mae_MX
where  mae_MX = mae_M + 2 · mae_C  for each split.

Including pseudotest in the objective directly penalises overfitting —
a trial that memorises the val fold scores poorly on pseudotest and Optuna
deprioritises it automatically.

Search space
────────────
    lr          LogU[1e-5, 5e-4]    — tighter range centred on 1e-4
    wd          LogU[1e-5, 1e-2]
    w_C         LogU[0.5, 5.0]      — small: C+ abundant, mainly balances
                                      backbone gradients vs w_M/w_X
    w_M         LogU[5, 150]        — known to need high values
    w_X         LogU[10, 400]       — ratio w_X/w_M left free
    alpha_M     LogU[0.1, 10]
    alpha_X     LogU[0.1, 10]
    monitor     {val_mae_M, val_mae_C}  — which metric drives early stopping

Architecture (all fixed)
────────────────────────
    n_blocks=5, base_channels=16, fc_hidden=256, head_hidden=64,
    kernel_size=5, dropout_ratio=0.45, pooling_type=gap, use_skip=True

Usage
─────
    python -m flare-forecast.optuna_search \\
        --config flare-forecast/default.yml \\
        --n_trials 40 \\
        --study_name poisson_training_search \\
        --storage sqlite:///optuna_training.db

    → writes flare-foreacast/best_config_poisson.yml
"""

import sys, os, argparse, math, copy, json
sys.path.append(os.getcwd())

import yaml
import numpy as np
import pandas as pd
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pathlib import Path

try:
    from comet_ml import Experiment as CometExperiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ..data.dataset import MagnetogramMultiDataModule
from ..models.poisson import FlexConvNetMulti, LitPoissonMulti

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Fixed architecture ────────────────────────────────────────────────────────
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
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _tss(y_true: np.ndarray, y_pred: np.ndarray, tau: float = 0.5) -> float:
    obs  = (y_true >= 0.5).astype(int)
    pred = (y_pred >= tau).astype(int)
    TP = int(np.sum((obs == 1) & (pred == 1)))
    TN = int(np.sum((obs == 0) & (pred == 0)))
    FP = int(np.sum((obs == 0) & (pred == 1)))
    FN = int(np.sum((obs == 1) & (pred == 0)))
    n_pos, n_neg = TP + FN, TN + FP
    TPR = TP / n_pos if n_pos > 0 else 0.0
    TNR = TN / n_neg if n_neg > 0 else 0.0
    return TPR + TNR - 1.0


def _eval_loader(trainer, lit_model, loader) -> dict:
    preds  = trainer.predict(model=lit_model, dataloaders=loader)
    yt_all = np.concatenate([b[1].cpu().float().numpy() for b in preds])
    yp_all = np.concatenate([b[2].cpu().float().numpy() for b in preds])
    out = {}
    for i, cls in enumerate(['C', 'M', 'X']):
        yt, yp = yt_all[:, i], yp_all[:, i]
        out[f'mae_{cls}'] = float(np.abs(yt - yp).mean())
        out[f'tss_{cls}'] = float(_tss(yt, yp))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, base_config: dict) -> float:
    cfg = copy.deepcopy(base_config)

    # ── Training hyperparameters ───────────────────────────────────────────
    lr      = trial.suggest_float('lr',      1e-5,  5e-4,  log=True)
    wd      = trial.suggest_float('wd',      1e-5,  1e-2,  log=True)
    w_C     = trial.suggest_float('w_C',     0.5,   20.0,   log=True)
    w_M     = trial.suggest_float('w_M',     5.0,   150.0, log=True)
    w_X     = trial.suggest_float('w_X',     10.0,  400.0, log=True)
    alpha_M = trial.suggest_float('alpha_M', 0.1,   10.0,  log=True)
    alpha_X = trial.suggest_float('alpha_X', 0.1,   10.0,  log=True)
    monitor = trial.suggest_categorical('monitor', ['val_mae_M', 'val_mae_C'])

    # ── Data ──────────────────────────────────────────────────────────────
    data = MagnetogramMultiDataModule(
        data_file       = cfg['data']['data_file'],
        target_cols     = cfg['data'].get('target_cols',
                                          ['C_count', 'M_count', 'X_count']),
        feature_cols    = cfg['data']['feature_cols'],
        val_split       = cfg['data']['val_split'],
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

    labels = data.train_set.label_frame
    mean_C, mean_M, mean_X = (float(labels[:, i].mean()) for i in range(3))

    max_epochs = cfg.get('optuna', {}).get('max_epochs_per_trial', 60)

    # ── Model (fixed architecture) ─────────────────────────────────────────
    backbone = FlexConvNetMulti(
        dim          = cfg['data']['dim'],
        length       = len(cfg['data']['channels']),
        len_features = len(cfg['data']['feature_cols']),
        n_blocks      = FIXED_ARCH['n_blocks'],
        base_channels = FIXED_ARCH['base_channels'],
        fc_hidden     = FIXED_ARCH['fc_hidden'],
        head_hidden   = FIXED_ARCH['head_hidden'],
        pooling_type  = FIXED_ARCH['pooling_type'],
        kernel_size   = FIXED_ARCH['kernel_size'],
        dropoutRatio  = FIXED_ARCH['dropout_ratio'],
        use_skip      = FIXED_ARCH['use_skip'],
        mean_C=mean_C, mean_M=mean_M, mean_X=mean_X,
    )
    lit_model = LitPoissonMulti(
        model=backbone, lr=lr, wd=wd, epochs=max_epochs,
        scheduler_type = cfg['training'].get('scheduler', 'cosine_warmup'),
        w_C=w_C, w_M=w_M, w_X=w_X,
        alpha_M=alpha_M, alpha_X=alpha_X,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator         = cfg['training']['device'],
        devices             = cfg['training'].get('devices', 1),
        max_epochs          = max_epochs,
        callbacks           = [
            EarlyStopping(monitor=monitor, min_delta=0.0005,
                          patience=12, mode='min',
                          strict=False, check_finite=False),
        ],
        logger              = False,
        enable_progress_bar = False,
        enable_model_summary= False,
        precision           = 32,
    )
    trainer.fit(model=lit_model, datamodule=data)

    # ── Evaluate ──────────────────────────────────────────────────────────
    val_m = _eval_loader(trainer, lit_model, data.val_dataloader())
    pt_m  = _eval_loader(trainer, lit_model, data.pseudotest_dataloader())

    val_obj = val_m['mae_C'] + 2.0 * val_m['mae_M']
    pt_obj  = pt_m['mae_C']  + 2.0 * pt_m['mae_M']
    total   = val_obj + pt_obj

    if not math.isfinite(total):
        return float('inf')

    trial.set_user_attr('val_mae_M',       val_m['mae_M'])
    trial.set_user_attr('val_mae_C',       val_m['mae_C'])
    trial.set_user_attr('val_tss_M',       val_m['tss_M'])
    trial.set_user_attr('val_tss_C',       val_m['tss_C'])
    trial.set_user_attr('pt_mae_M',        pt_m['mae_M'])
    trial.set_user_attr('pt_mae_C',        pt_m['mae_C'])
    trial.set_user_attr('pt_tss_M',        pt_m['tss_M'])
    trial.set_user_attr('pt_tss_C',        pt_m['tss_C'])
    trial.set_user_attr('val_obj',         val_obj)
    trial.set_user_attr('pt_obj',          pt_obj)
    trial.set_user_attr('monitor_used',    monitor)
    trial.set_user_attr('overfitting_gap',
                        abs(val_m['mae_M'] - pt_m['mae_M'])
                        + abs(val_m['mae_C'] - pt_m['mae_C']))

    print(f"  trial {trial.number:3d}  obj={total:.4f}  "
          f"mon={monitor.split('_')[-1]}  "
          f"lr={lr:.2e}  w_C={w_C:.2f}  w_M={w_M:.1f}  w_X={w_X:.1f}  "
          f"αM={alpha_M:.2f}  αX={alpha_X:.2f}  "
          f"[val] mae_M={val_m['mae_M']:.4f} mae_C={val_m['mae_C']:.4f} "
          f"tss_M={val_m['tss_M']:.3f}  "
          f"[pt]  mae_M={pt_m['mae_M']:.4f} tss_M={pt_m['tss_M']:.3f}")

    return total


# ─────────────────────────────────────────────────────────────────────────────
# COMET SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _log_search_to_comet(study: optuna.Study, comet_cfg: dict, config: dict):
    if not COMET_AVAILABLE or not comet_cfg.get('enabled', False):
        return

    exp = CometExperiment(
        api_key      = comet_cfg['api_key'],
        project_name = comet_cfg.get('optuna_project_name',
                                     comet_cfg['project_name'] + '-optuna'),
        workspace    = comet_cfg.get('workspace'),
    )
    exp.set_name(f"optuna_training_{study.study_name}")
    exp.log_parameters(FIXED_ARCH)

    best = study.best_trial
    exp.log_parameters(best.params)
    exp.log_metric('best_objective',     best.value)
    exp.log_metric('best_trial_number',  best.number)
    exp.log_metric('n_trials_completed', len(study.trials))
    for k, v in best.user_attrs.items():
        exp.log_metric(f'best_{k}', v)

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {'trial': t.number, 'objective': t.value}
        row.update(t.params)
        row.update({f'diag_{k}': v for k, v in t.user_attrs.items()})
        rows.append(row)
    if rows:
        exp.log_asset_data(pd.DataFrame(rows).to_csv(index=False),
                           name='all_trials.csv')

    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            exp.log_metric('trial_objective', t.value, step=t.number)

    exp.log_other('fixed_architecture', json.dumps(FIXED_ARCH))
    exp.end()
    print(f"\nOptuna summary → Comet: {exp.url}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='flare-foreacast/config.yml')
    parser.add_argument('--n_trials',   type=int, default=40)
    parser.add_argument('--study_name', default='poisson_training_search')
    parser.add_argument('--storage',    default=None,
                        help='e.g. sqlite:///optuna_training.db')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config['data']['test'] = config['data']['test'].lower()

    pl.seed_everything(args.seed, workers=True)

    print(f"\n{'='*60}")
    print(f"  Fixed architecture:")
    for k, v in FIXED_ARCH.items():
        print(f"    {k:20s} = {v}")
    print(f"{'='*60}\n")

    study = optuna.create_study(
        study_name     = args.study_name,
        direction      = 'minimize',
        sampler        = optuna.samplers.TPESampler(seed=args.seed),
        pruner         = optuna.pruners.MedianPruner(n_startup_trials=5,
                                                     n_warmup_steps=5),
        storage        = args.storage,
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: objective(trial, config),
        n_trials          = args.n_trials,
        show_progress_bar = True,
    )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Best trial #{best.number}  →  objective = {best.value:.6f}")
    print(f"{'='*60}")
    for k, v in best.params.items():
        print(f"  {k:20s} = {v}")
    print(f"\n  Diagnostics:")
    for k, v in best.user_attrs.items():
        print(f"  {k:25s} = {v:.4f}" if isinstance(v, float) else
              f"  {k:25s} = {v}")

    # ── Save best config ──────────────────────────────────────────────────
    best_cfg = copy.deepcopy(config)
    best_cfg['model'].update({
        'n_blocks':      FIXED_ARCH['n_blocks'],
        'base_channels': FIXED_ARCH['base_channels'],
        'fc_hidden':     FIXED_ARCH['fc_hidden'],
        'head_hidden':   FIXED_ARCH['head_hidden'],
        'pooling_type':  FIXED_ARCH['pooling_type'],
        'kernel_size':   FIXED_ARCH['kernel_size'],
        'dropout_ratio': FIXED_ARCH['dropout_ratio'],
        'use_skip':      FIXED_ARCH['use_skip'],
    })
    p = best.params
    best_cfg['training'].update({
        'lr':             round(p['lr'],      10),
        'wd':             round(p['wd'],      10),
        'w_C':            round(p['w_C'],     4),
        'w_M':            round(p['w_M'],     4),
        'w_X':            round(p['w_X'],     4),
        'alpha_M':        round(p['alpha_M'], 4),
        'alpha_X':        round(p['alpha_X'], 4),
        'monitor_metric': p['monitor'],
    })
    best_cfg['optuna'] = {
        'search_type': 'training_only',
        'best_trial':  int(best.number),
        'best_value':  float(best.value),
        'n_trials':    args.n_trials,
        'storage':     args.storage,
        'objective': 'val_mae_CM + pseudotest_mae_CM  (mae_C + 2*mae_M)',
        'fixed_arch':  FIXED_ARCH,
    }

    out_path = Path(args.config).parent / 'best_config_poisson.yml'
    with open(out_path, 'w') as f:
        yaml.dump(best_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest config saved → {out_path}")

    _log_search_to_comet(study, config.get('comet', {}), config)


if __name__ == '__main__':
    main()