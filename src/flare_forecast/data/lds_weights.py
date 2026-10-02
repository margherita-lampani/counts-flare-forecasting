"""
Label Distribution Smoothing (LDS) utilities for imbalanced flare-count prediction.

This module implements Label Distribution Smoothing (LDS) following the
method proposed by Yang et al. (2021) for deep imbalanced regression.
Instead of weighting samples using the raw empirical target distribution,
the target histogram is smoothed with a symmetric kernel to estimate an
effective label density.

Main features:
    - Gaussian and Laplace smoothing kernels
    - Effective label-density estimation via histogram convolution
    - Inverse-density and square-root inverse-density reweighting
    - Weight normalization for stable optimization
    - Maximum-weight clipping to prevent extreme gradients

The resulting sample weights can be used during training to mitigate
the strong imbalance typically present in flare-count datasets, where
high-activity events are substantially rarer than quiet observations.

Reference:
    Yang, Y. et al. (2021)
    "Delving into Deep Imbalanced Regression"
    https://arxiv.org/abs/2102.09554
"""


import numpy as np
from scipy.ndimage import convolve1d
from scipy.stats import norm

def get_lds_kernel_window(kernel_type: str, ks: int, sigma: float):
    """
    Create the LDS kernel window.
    ks: kernel size (number of bins over which to spread the influence)
    sigma: kernel width
    """
    assert ks % 2 == 1, "kernel size must be odd"
    half_ks = (ks - 1) // 2
    
    if kernel_type == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = norm.pdf(np.arange(-half_ks, half_ks + 1), 0, sigma)
        kernel_window /= kernel_window.sum()  # normalize
        
    elif kernel_type == 'laplace':
        laplace = lambda x, sigma: np.exp(-abs(x) / sigma) / (2 * sigma)
        kernel_window = np.array([laplace(x, sigma) for x in range(-half_ks, half_ks + 1)])
        kernel_window /= kernel_window.sum()
        
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return kernel_window


def compute_lds_weights(labels: np.ndarray,
                        n_bins: int = 100,
                        kernel_type: str = 'gaussian',
                        ks: int = 5,
                        sigma: float = 2.0,
                        reweight: str = 'inverse',
                        max_weight_clip: float = 100.0) -> np.ndarray:
    """
    Compute LDS weights for each sample in the training set.
    
    Args:
        labels:           1D array of targets (in normalized or log space)
        n_bins:           number of bins for the histogram
        kernel_type:      'gaussian' or 'laplace'
        ks:               kernel size (must be odd)
        sigma:            std of the Gaussian kernel (in bin units)
        reweight:         'inverse' → 1/p̃, 'sqrt_inverse' → 1/√p̃
        max_weight_clip:  maximum weight cap (avoids huge weights for empty bins)
    
    Returns:
        weights:          1D array of weights, one per sample
    """
    # 1. Empirical histogram
    hist, bin_edges = np.histogram(labels, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 2. Convolution with symmetric kernel → effective density
    kernel_window = get_lds_kernel_window(kernel_type, ks, sigma)
    eff_density = convolve1d(hist.astype(float), weights=kernel_window, mode='reflect')
    
    # Clamp: avoid division by zero in completely empty bins
    eff_density = np.maximum(eff_density, 1e-6)
    
    # 3. Compute weight per bin
    if reweight == 'inverse':
        bin_weights = 1.0 / eff_density
    elif reweight == 'sqrt_inverse':
        bin_weights = 1.0 / np.sqrt(eff_density)
    else:
        raise ValueError(f"reweight must be 'inverse' or 'sqrt_inverse'")
    
    # Normalize: mean weight = 1 (does not change the learning rate scale)
    bin_weights = bin_weights / bin_weights.mean()
    
    # 4. Clip maximum weights for stability
    bin_weights = np.clip(bin_weights, 0, max_weight_clip)
    
    # 5. Map each sample to its bin
    bin_idxs = np.searchsorted(bin_edges[1:-1], labels, side='right')
    bin_idxs = np.clip(bin_idxs, 0, n_bins - 1)
    
    sample_weights = bin_weights[bin_idxs]
    return sample_weights.astype(np.float32)
