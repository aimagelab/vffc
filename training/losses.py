import torch
import torch.nn as nn

from .BCEDiceLoss import BCEDiceLoss
from .dice import soft_dice_loss


def get_criterion(loss: str, weigh_bce: float = 1, device: torch.device = torch.device('cpu')):

    if loss == 'bce':
        if weigh_bce != 1:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weigh_bce])).to(device)
        else:
            return nn.BCEWithLogitsLoss().to(device)
    if loss == 'dice':
        return soft_dice_loss
    if loss == 'bce_dice':
        return BCEDiceLoss(weigh_bce=weigh_bce).to(device)

    raise ValueError(f'Unknown loss {loss}')

