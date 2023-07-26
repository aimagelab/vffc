import torch
import torch.nn.functional as F


def soft_dice_loss(y_pred, target, smooth=0.0, eps=1e-7, dims=None):
    assert y_pred.size() == target.size()
    mode = 'BINARY' if len(target.shape) == 3 else 'MULTICLASS'
    if mode == 'BINARY':
        y_pred = F.logsigmoid(y_pred).exp()
    else:
        raise NotImplementedError(f'Mode {mode} not implemented')

    if dims is not None:
        intersection = torch.sum(y_pred * target, dim=dims)
        cardinality = torch.sum(y_pred + target, dim=dims)
    else:
        intersection = torch.sum(y_pred * target)
        cardinality = torch.sum(y_pred + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    dice_loss = 1 - dice_score
    return dice_loss.mean()
