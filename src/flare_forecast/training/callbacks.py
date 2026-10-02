"""
Custom PyTorch Lightning callbacks and verification metrics for solar flare forecasting.

This module implements forecast-verification metrics commonly used in
space-weather prediction, including the True Skill Statistic (TSS) and
Heidke Skill Score (HSS2). These metrics are computed for C-class, M-class,
and X-class flare occurrence and are automatically logged during model
validation.

Main features:
    - Binary forecast verification for C+, M+, and X+ flare occurrence
    - Computation of confusion matrices and derived skill scores
    - Automatic logging during validation epochs
    - Support for both Poisson-regression and binary-classification models
    - Configurable primary forecast class for model selection

Implemented metrics:
    - True Skill Statistic (TSS)
    - Heidke Skill Score (HSS2)
    - True Positive Rate (TPR)
    - True Negative Rate (TNR)
    - Full confusion-matrix statistics

These metrics provide operationally relevant measures of forecasting
performance and are used to monitor training progress and select
optimal model checkpoints.
"""


import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

OUTPUT_IDX = {'C': 0, 'M': 1, 'X': 2}


def confusion_skills(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Compute TP/TN/FP/FN, TPR, TNR, TSS, HSS2 for binary arrays."""
    TP = int(np.sum((obs == 1) & (pred == 1)))
    TN = int(np.sum((obs == 0) & (pred == 0)))
    FP = int(np.sum((obs == 0) & (pred == 1)))
    FN = int(np.sum((obs == 1) & (pred == 0)))
    n_pos, n_neg = TP + FN, TN + FP
    TPR  = TP / n_pos if n_pos > 0 else 0.0
    TNR  = TN / n_neg if n_neg > 0 else 0.0
    TSS  = TPR + TNR - 1.0
    denom = (n_pos * (FN + TN)) + ((TP + FP) * n_neg)
    HSS2 = (2 * (TP * TN - FP * FN)) / denom if denom > 0 else 0.0
    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                n_pos=n_pos, n_neg=n_neg,
                TPR=TPR, TNR=TNR, TSS=TSS, HSS2=HSS2)


class TSSCallback(Callback):
    """
    Accumulates validation predictions and computes TSS / HSS2 for C, M, X.

    Parameters
    ----------
    primary_key : 'C' | 'M' | 'X'
        Class whose TSS is also exposed as ``val_TSS_primary``. Default: 'M'.
    thresh : float
        Decision threshold applied to model output. Default: 0.5.
    mode : 'poisson' | 'binary'
        Shown in the diagnostic print-out only; logic is identical.
    """

    def __init__(self, primary_key: str = 'M',
                 thresh: float = 0.5,
                 mode: str = 'poisson'):
        super().__init__()
        if primary_key not in OUTPUT_IDX:
            raise ValueError(f"primary_key must be one of {list(OUTPUT_IDX)}")
        self.primary_key = primary_key
        self.thresh      = thresh
        self.mode        = mode
        self._reset()

    def _reset(self):
        self._y_true: list = []
        self._y_pred: list = []

    def on_validation_batch_end(self, trainer, pl_module, outputs,
                                batch, batch_idx, dataloader_idx=0):
        _, x, f, y = batch
        with torch.no_grad():
            out = pl_module.model(x, f)
        self._y_true.append(y.detach().cpu().float().numpy())
        self._y_pred.append(out.detach().cpu().float().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._y_true:
            return

        y_true = np.concatenate(self._y_true, axis=0)
        y_pred = np.concatenate(self._y_pred, axis=0)

        log_dict = {}
        for name, idx in OUTPUT_IDX.items():
            obs  = (y_true[:, idx] >= 0.5).astype(int)
            pred = (y_pred[:, idx] >= self.thresh).astype(int)
            r    = confusion_skills(obs, pred)

            log_dict[f'val_TSS_{name}']  = float(r['TSS'])
            log_dict[f'val_HSS2_{name}'] = float(r['HSS2'])
            if name == self.primary_key:
                log_dict['val_TSS_primary'] = float(r['TSS'])

            print(f"  [TSS | {self.mode}] {name}+  "
                  f"TSS={r['TSS']:.4f}  HSS2={r['HSS2']:.4f}  "
                  f"TP={r['TP']}  TN={r['TN']}  FP={r['FP']}  FN={r['FN']}  "
                  f"TPR={r['TPR']:.3f}  TNR={r['TNR']:.3f}  "
                  f"(pos={r['n_pos']}  neg={r['n_neg']})")

        pl_module.log_dict(log_dict, on_epoch=True, prog_bar=False)
        self._reset()