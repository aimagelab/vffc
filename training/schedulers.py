from torch import optim


def get_lr_scheduler(scheduler: str, lr: float, optimizer: optim.Optimizer, steps_per_epoch: int, epochs: int):
    if scheduler == 'one_cycle':
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
    elif scheduler == 'constant':
        lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, lr)
    else:
        raise ValueError(f'Unknown scheduler {scheduler}')

    return lr_scheduler
