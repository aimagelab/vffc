import torch
import torch.nn as nn

from .dice import soft_dice_loss


class BCEDiceLoss(nn.Module):
    def __init__(self, weigh_bce=1, smooth=0.0, eps=1e-7, dims=None):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps
        self.dims = dims

        if weigh_bce != 1:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weigh_bce]))
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        bce_logits_loss = self.bce(input=output, target=target)
        dice_loss = soft_dice_loss(output, target, self.smooth, self.eps, self.dims)
        return 0.5 * bce_logits_loss + 0.5 * dice_loss
