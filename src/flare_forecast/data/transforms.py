"""
Custom data augmentation transforms for solar magnetogram forecasting.

This module implements domain-specific augmentations designed for magnetic
field observations. In particular, it provides random polarity inversion,
which exploits the sign-symmetry of solar magnetic fields by flipping the
magnetogram intensity values while preserving the underlying physical
structure of the active region.

Implemented transforms:
    - RandomPolaritySwitch: randomly multiplies the magnetogram by -1
      with a configurable probability.

These augmentations can be integrated into the training pipeline to improve
model robustness and reduce sensitivity to magnetic-polarity orientation.
"""

from torch import nn
import torch

class RandomPolaritySwitch(torch.nn.Module):
    """Inverts the polarity of the given magnetogram randomly with a given probability.
    If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.

    Args:
        p (float): probability of the image being color inverted. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (Tensor): Image to be inverted.

        Returns:
            Tensor: Randomly polarity inverted image.
        """
        if torch.rand(1).item() < self.p:
            return -img
        
        return img


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"