"""
Multi-task Poisson CNN for solar flare count forecasting.

This module implements the primary forecasting architecture used in the
project. The model jointly predicts the expected number of future
C-class, M-class, and X-class flares from solar active-region
magnetograms and optional scalar magnetic features.

Model characteristics:
    - Shared convolutional feature extractor
    - Independent Poisson output heads for C, M, and X flare counts
    - Optional fusion of image-based and tabular magnetic features
    - Residual connections and configurable architecture depth
    - Positive rate predictions enforced through Softplus activations

Outputs:
    λ_C : expected number of C-class flares
    λ_M : expected number of M-class flares
    λ_X : expected number of X-class flares

Training objective:
    The model is optimized using a weighted multi-task Poisson
    negative log-likelihood loss. Additional binary detection terms
    can be included for rare M-class and X-class events to improve
    sensitivity under severe class imbalance.

Initialization:
    Output-layer biases are initialized from the empirical mean flare
    counts observed in the training data, providing physically meaningful
    starting predictions and faster convergence.

The module includes both the neural-network architecture
(FlexConvNetMulti) and its PyTorch Lightning training wrapper
(LitPoissonMulti).
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

class FlexConvNetMulti(nn.Module):
    def __init__(self, dim=128, length=1, len_features=0,
                 n_blocks=4, base_channels=16, fc_hidden=64,
                 head_hidden=32, pooling_type='gap',
                 kernel_size=3, dropoutRatio=0.0, use_skip=False,
                 mean_C=None, mean_M=None, mean_X=None, weights=None):
        super().__init__()
        self.len_features = len_features
        self.pooling_type = pooling_type

        conv_blocks, skip_blocks = [], []
        in_ch, out_ch = length, base_channels
        for i in range(n_blocks):
            pad = kernel_size // 2
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
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

        fc_in = (nn.Linear(final_ch, fc_hidden)
                 if pooling_type == 'gap' else nn.LazyLinear(fc_hidden))
        self.shared_fc = nn.Sequential(
            fc_in, nn.ReLU(inplace=True), nn.Dropout(dropoutRatio))

        if len_features > 0:
            self.feature_fc = nn.Sequential(
                nn.Linear(fc_hidden + len_features, fc_hidden),
                nn.ReLU(inplace=True), nn.Dropout(dropoutRatio))

        def make_head(hidden):
            hidden = max(hidden, 8)
            return nn.Sequential(
                nn.Linear(fc_hidden, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, 1), nn.Softplus())

        self.head_C = make_head(head_hidden)
        self.head_M = make_head(head_hidden)
        self.head_X = make_head(head_hidden)

        if pooling_type == 'flatten':
            self.forward(torch.zeros(1, length, dim, dim * 2),
                         torch.zeros(1, max(len_features, 1)))

        self.apply(self._init_weights)

        for head, mean_val in [(self.head_C, mean_C),
                               (self.head_M, mean_M),
                               (self.head_X, mean_X)]:
            if mean_val is not None and mean_val > 0:
                with torch.no_grad():
                    head[-2].bias.fill_(math.log(max(mean_val, 1e-3)))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight);  nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear, nn.LazyLinear)):
            nn.init.xavier_normal_(m.weight);   nn.init.zeros_(m.bias)

    def forward(self, x, f):
        for conv, skip in zip(self.conv_blocks, self.skip_blocks):
            x = conv(x);  x = skip(x)
        x = x.mean(dim=(-2, -1)) if self.pooling_type == 'gap' else x.view(x.size(0), -1)
        x = self.shared_fc(x)
        if self.len_features > 0:
            x = self.feature_fc(torch.cat([x, f], dim=1))
        return torch.cat([self.head_C(x), self.head_M(x), self.head_X(x)], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

def poisson_bce_loss(lam, y, w_C=1.0, w_M=5.0, w_X=20.0,
                     alpha_M=1.0, alpha_X=2.0, eps=1e-7):
    def nll(lam_k, y_k):
        return (lam_k - y_k * torch.log(lam_k.clamp(min=eps))).mean()

    loss = w_C * nll(lam[:,0], y[:,0]) \
         + w_M * nll(lam[:,1], y[:,1]) \
         + w_X * nll(lam[:,2], y[:,2])

    def bce_p(lam_k, y_k):
        t = (y_k >= 1.0).float()
        log_p = torch.log1p(-torch.exp(-lam_k).clamp(max=1.0 - eps))
        return (-(t * log_p + (1.0 - t) * (-lam_k))).mean()

    if alpha_M > 0: loss = loss + alpha_M * bce_p(lam[:,1], y[:,1])
    if alpha_X > 0: loss = loss + alpha_X * bce_p(lam[:,2], y[:,2])
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTNING MODULE
# ─────────────────────────────────────────────────────────────────────────────

class LitPoissonMulti(pl.LightningModule):
    """
    Lightning wrapper for FlexConvNetMulti.

    Metrics logged each epoch
    ─────────────────────────
    {phase}_mae_C/M/X     MAE per class
    {phase}_mae_mean      unweighted mean MAE (C, M, X)
    {phase}_mae_MX        mae_M + 2·mae_X  
    val_nll_C/M/X         Poisson NLL per class (diagnostic)
    val_loss              total weighted loss
    """

    CLASS_NAMES = ['C', 'M', 'X']

    def __init__(self, model, lr=1e-4, wd=1e-2, epochs=100,
                 scheduler_type='cosine_warmup',
                 w_C=1.0, w_M=5.0, w_X=20.0,
                 alpha_M=1.0, alpha_X=2.0, eps=1e-7):
        super().__init__()
        self.model = model
        self.lr = lr;  self.wd = wd;  self.epochs = epochs
        self.scheduler_type = scheduler_type
        self.w_C = w_C;  self.w_M = w_M;  self.w_X = w_X
        self.alpha_M = alpha_M;  self.alpha_X = alpha_X
        self.eps = eps
        for phase in ('train', 'val'):
            for cls in self.CLASS_NAMES:
                setattr(self, f'{phase}_mae_{cls}',
                        torchmetrics.MeanAbsoluteError())

    def _loss(self, lam, y):
        return poisson_bce_loss(lam, y, w_C=self.w_C, w_M=self.w_M,
                                w_X=self.w_X, alpha_M=self.alpha_M,
                                alpha_X=self.alpha_X, eps=self.eps)

    def _nll_per_class(self, lam, y):
        return {cls: (lam[:,i] - y[:,i]*torch.log(lam[:,i].clamp(min=self.eps))).mean()
                for i, cls in enumerate(self.CLASS_NAMES)}

    def _log_mae(self, phase, lam, y):
        log_dict = {}
        for i, cls in enumerate(self.CLASS_NAMES):
            m = getattr(self, f'{phase}_mae_{cls}')
            m(lam[:,i], y[:,i])
            log_dict[f'{phase}_mae_{cls}'] = m

        # Unweighted mean
        log_dict[f'{phase}_mae_mean'] = sum(
            lam[:,i].sub(y[:,i]).abs().mean() for i in range(N_CLASSES)) / N_CLASSES

        # ── Composite metric: mae_M + 2·mae_X ────────────────────────────
        # Recommended as monitor_metric in config.yml.
        # Rationale: val_mae_X alone is too noisy (99.7% of samples have
        # X=0, so the minimum often corresponds to λ_X≈0 everywhere).
        # val_mae_MX keeps focus on both rare classes and is more stable.
        log_dict[f'{phase}_mae_MX'] = (
            lam[:,1].sub(y[:,1]).abs().mean()
            + 2.0 * lam[:,2].sub(y[:,2]).abs().mean()
        )

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False)

    def training_step(self, batch, batch_idx):
        fname, x, f, y = batch;  y = y.float()
        lam = self.model(x, f)
        loss = self._loss(lam, y)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({f'train_nll_{c}': v
                       for c, v in self._nll_per_class(lam.detach(), y.detach()).items()},
                      on_step=False, on_epoch=True)
        self._log_mae('train', lam.detach(), y.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        fname, x, f, y = batch;  y = y.float()
        lam = self.model(x, f)
        self.log('val_loss', self._loss(lam, y),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({f'val_nll_{c}': v
                       for c, v in self._nll_per_class(lam, y).items()},
                      on_step=False, on_epoch=True)
        self._log_mae('val', lam, y)

    def test_step(self, batch, batch_idx):
        fname, x, f, y = batch
        self.log('test_loss', self._loss(self.model(x, f), y.float()),
                 on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        fname, x, f, y = batch
        return fname, y.float(), self.model(x, f)

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.scheduler_type == 'cosine_warmup':
            warmup = max(1, int(self.epochs * 0.10))
            def lr_lambda(epoch):
                if epoch < warmup: return (epoch + 1) / warmup
                return 0.5 * (1.0 + math.cos(
                    math.pi * (epoch - warmup) / max(1, self.epochs - warmup)))
            sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        else:
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return [opt], [sched]

    def on_load_checkpoint(self, checkpoint):
        sd, msd = checkpoint['state_dict'], self.state_dict()
        for k in sd:
            if k in msd and sd[k].shape != msd[k].shape:
                sd[k] = msd[k]
        checkpoint.pop('optimizer_states', None)