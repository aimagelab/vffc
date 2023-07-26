import torch.nn as nn
from torch import optim
from utils.logger import get_logger

logger = get_logger(__file__)


def get_optimizer(optimizer: str, model: nn.Module, lr: float):
    params = model.parameters()
    if optimizer == 'adamW':
        return optim.AdamW(params, lr=lr)
    if optimizer == 'sgd':
        return optim.SGD(params, lr=lr)
    raise ValueError(f'Unknown optimizer {optimizer}')
