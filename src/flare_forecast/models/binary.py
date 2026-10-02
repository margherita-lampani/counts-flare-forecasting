"""
Multi-task CNN for binary solar flare occurrence forecasting.

This module implements a convolutional neural network that predicts the
probability of future C-class, M-class, and X-class flare occurrence from
solar active-region magnetograms and optional scalar magnetic features.

Model characteristics:
    - Shared convolutional feature extractor
    - Three independent binary classification heads
    - Sigmoid probability outputs for C, M, and X flare occurrence
    - Optional magnetic-feature fusion in the fully connected layers
    - Residual connections and configurable architecture depth

Labels:
    Binary targets are generated on-the-fly from flare-count labels:

        t_k = 1  if count_k >= 1
        t_k = 0  otherwise

Training objective:
    Weighted binary cross-entropy is used to mitigate the strong class
    imbalance present in flare forecasting datasets, with independent
    class weights for C-, M-, and X-class events.

Calibration:
    Decision thresholds are optimized after training by maximizing the
    True Skill Statistic (TSS) on the validation set.

Initialization:
    Output-layer biases are initialized from the observed training-set
    flare frequencies using log-odds priors, providing calibrated starting
    probabilities and faster convergence.

Outputs:
    [P(C≥1), P(M≥1), P(X≥1)]
"""

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torch import optim

from flare_forecast.models.blocks import ResidualBlock

N_CLASSES = 3


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE
# ─────────────────────────────────────────────────────────────────────────────

class FlexConvNetBinary(nn.Module):
    """
    Convolutional backbone with three independent sigmoid output heads.
    Architecture is identical to FlexConvNetMulti; only the final activation
    changes (Sigmoid instead of Softplus).

    Parameters
    ----------
    dim, length, len_features, n_blocks, base_channels, fc_hidden,
    head_hidden, pooling_type, kernel_size, dropoutRatio, use_skip
        Identical meaning to FlexConvNetMulti.
    pos_rate_C, pos_rate_M, pos_rate_X : float | None
        Training-set positive rates for smart bias initialisation.
    """

    def __init__(self, dim: int = 128, length: int = 1, len_features: int = 0,
                 n_blocks: int = 4, base_channels: int = 16, fc_hidden: int = 64,
                 head_hidden: int = 32, pooling_type: str = 'gap',
                 kernel_size: int = 3, dropoutRatio: float = 0.0,
                 use_skip: bool = False,
                 pos_rate_C: float = None,
                 pos_rate_M: float = None,
                 pos_rate_X: float = None):
        super().__init__()
        self.len_features = len_features
        self.pooling_type = pooling_type

        # ── Conv stem ─────────────────────────────────────────────────────
        conv_blocks, skip_blocks = [], []
        in_ch, out_ch = length, base_channels
        for i in range(n_blocks):
            pad   = kernel_size // 2
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            if i < n_blocks - 1:
                block.add_module('pool', nn.MaxPool2d(2, stride=2))
            conv_blocks.append(block)
            skip_blocks.append(
                ResidualBlock(out_ch, kernel_size)
                if use_skip and i < n_blocks - 1 else nn.Identity()
            )
            in_ch, out_ch = out_ch, min(out_ch * 2, 512)

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.skip_blocks = nn.ModuleList(skip_blocks)
        final_ch = in_ch

        # ── Shared FC trunk ───────────────────────────────────────────────
        fc_in = (nn.Linear(final_ch, fc_hidden)
                 if pooling_type == 'gap' else nn.LazyLinear(fc_hidden))
        self.shared_fc = nn.Sequential(fc_in, nn.ReLU(inplace=True),
                                       nn.Dropout(dropoutRatio))
        if len_features > 0:
            self.feature_fc = nn.Sequential(
                nn.Linear(fc_hidden + len_features, fc_hidden),
                nn.ReLU(inplace=True), nn.Dropout(dropoutRatio),
            )

        # ── Three binary heads ────────────────────────────────────────────
        def _head(hidden):
            h = max(hidden, 8)
            return nn.Sequential(
                nn.Linear(fc_hidden, h), nn.ReLU(inplace=True),
                nn.Linear(h, 1),         # bias init target (index -2)
                nn.Sigmoid(),
            )

        self.head_C = _head(head_hidden)
        self.head_M = _head(head_hidden)
        self.head_X = _head(head_hidden)

        if pooling_type == 'flatten':
            self.forward(torch.zeros(1, length, dim, dim * 2),
                         torch.zeros(1, max(len_features, 1)))

        self.apply(self._init_weights)

        # Smart bias init: log-odds of base positive rate
        for head, rate in [(self.head_C, pos_rate_C),
                           (self.head_M, pos_rate_M),
                           (self.head_X, pos_rate_X)]:
            if rate is not None and 0.0 < rate < 1.0:
                with torch.no_grad():
                    head[-2].bias.fill_(math.log(rate / (1.0 - rate)))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight);  nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear, nn.LazyLinear)):
            nn.init.xavier_normal_(m.weight);   nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        for conv, skip in zip(self.conv_blocks, self.skip_blocks):
            x = conv(x);  x = skip(x)
        x = (x.mean(dim=(-2, -1)) if self.pooling_type == 'gap'
             else x.view(x.size(0), -1))
        x = self.shared_fc(x)
        if self.len_features > 0:
            x = self.feature_fc(torch.cat([x, f], dim=1))
        return torch.cat([self.head_C(x), self.head_M(x), self.head_X(x)], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

def weighted_bce_loss(p: torch.Tensor, y: torch.Tensor,
                      w_C: float = 1.0, w_M: float = 5.0, w_X: float = 20.0,
                      eps: float = 1e-7) -> torch.Tensor:
    """Weighted BCE; labels binarised on-the-fly: t_k = 1 if y_k >= 1."""
    def bce(p_k, y_k):
        t = (y_k >= 1.0).float()
        return (-(t * torch.log(p_k.clamp(min=eps))
                  + (1.0 - t) * torch.log((1.0 - p_k).clamp(min=eps)))).mean()
    return (w_C * bce(p[:, 0], y[:, 0])
            + w_M * bce(p[:, 1], y[:, 1])
            + w_X * bce(p[:, 2], y[:, 2]))


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTNING MODULE
# ─────────────────────────────────────────────────────────────────────────────

class LitBinary(pl.LightningModule):
    """Lightning wrapper for FlexConvNetBinary with weighted BCE loss."""

    CLASS_NAMES = ['C', 'M', 'X']

    def __init__(self, model: nn.Module,
                 lr: float = 1e-4, wd: float = 1e-2, epochs: int = 100,
                 scheduler_type: str = 'cosine_warmup',
                 w_C: float = 1.0, w_M: float = 5.0, w_X: float = 20.0,
                 eps: float = 1e-7):
        super().__init__()
        self.model = model
        self.lr = lr;  self.wd = wd;  self.epochs = epochs
        self.scheduler_type = scheduler_type
        self.w_C = w_C;  self.w_M = w_M;  self.w_X = w_X
        self.eps = eps
        for cls in self.CLASS_NAMES:
            setattr(self, f'val_auroc_{cls}', torchmetrics.AUROC(task='binary'))

    def _loss(self, p, y):
        return weighted_bce_loss(p, y,
                                 w_C=self.w_C, w_M=self.w_M, w_X=self.w_X,
                                 eps=self.eps)

    def _log_auroc(self, p, y):
        log_dict = {}
        for i, cls in enumerate(self.CLASS_NAMES):
            t = (y[:, i] >= 1.0).long()
            if t.sum() > 0 and (1 - t).sum() > 0:
                m = getattr(self, f'val_auroc_{cls}')
                m(p[:, i], t)
                log_dict[f'val_auroc_{cls}'] = m
        if log_dict:
            self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False)

    def training_step(self, batch, batch_idx):
        _, x, f, y = batch
        loss = self._loss(self.model(x, f), y.float())
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, f, y = batch;  y = y.float()
        p = self.model(x, f)
        self.log('val_loss', self._loss(p, y),
                 on_step=False, on_epoch=True, prog_bar=True)
        self._log_auroc(p, y)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        fname, x, f, y = batch
        return fname, y.float(), self.model(x, f)

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.scheduler_type == 'cosine_warmup':
            w = max(1, int(self.epochs * 0.10))
            lam = lambda e: ((e+1)/w if e < w else
                             0.5*(1+math.cos(math.pi*(e-w)/max(1,self.epochs-w))))
            sched = optim.lr_scheduler.LambdaLR(opt, lam)
        else:
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return [opt], [sched]

    def on_load_checkpoint(self, checkpoint):
        sd, msd = checkpoint['state_dict'], self.state_dict()
        for k in sd:
            if k in msd and sd[k].shape != msd[k].shape:
                sd[k] = msd[k]
        checkpoint.pop('optimizer_states', None)