"""Asymmetric Loss (ASL) for multi-label audio classification.

Reference:
    Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.
    Addresses extreme negative-positive imbalance and suppresses false positive over-firing
    by applying asymmetric focusing (gamma_neg > gamma_pos) and probability margin clipping.
"""

from __future__ import annotations

import torch
from torch import nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for Multi-Label Classification with Logits input.

    Parameters:
        gamma_neg: Focusing parameter for negative examples (default 4.0).
                   Down-weights easy negatives so they don't produce noisy gradients.
        gamma_pos: Focusing parameter for positive examples (default 1.0).
        clip: Margin threshold for negative examples (default 0.05).
              Negative probabilities below this margin produce exactly zero loss,
              allowing the model to output confident zeros on quiet/ambient sound.
        eps: Small epsilon for numerical stability.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute ASL loss.

        Args:
            x: Logits tensor of shape (batch_size, num_classes).
            y: Binary targets tensor of shape (batch_size, num_classes) with values in {0, 1}.

        Returns:
            Scalar loss tensor.
        """
        # Compute probabilities safely
        probs = torch.sigmoid(x)
        p_pos = probs
        p_neg = 1.0 - probs

        # Apply asymmetric margin to negatives
        if self.clip is not None and self.clip > 0.0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)

        # Basic binary cross entropy components
        loss_pos = y * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = (1.0 - y) * torch.log(p_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # Asymmetric focusing
        if self.gamma_neg > 0.0 or self.gamma_pos > 0.0:
            p_target = p_pos * y + p_neg * (1.0 - y)
            gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            weight = torch.pow(1.0 - p_target, gamma)
            loss = loss * weight

        return -loss.sum() / x.size(0)
