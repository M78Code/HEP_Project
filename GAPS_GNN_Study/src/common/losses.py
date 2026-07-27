"""Loss functions used by the GAPS GNN training scripts."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for binary or multi-class classification.

    Args:
        gamma   : focusing parameter. gamma=0 is equivalent to cross entropy.
        alpha   : optional class weights. None means no class weighting.
        reduction : 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 1.5, alpha=None, reduction: str='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : model output before softmax, [batch_size, num_classes].
            targets : integer class labels, [batch_size].
        Returns:
            loss according to reduction.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
