import torch.utils.data
import torch

from utils.dist import is_distributed

def get_samplers(train_dataset, eval_dataset, seed, generator):
    if is_distributed():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=seed)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, seed=seed)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
        eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
    return train_sampler, eval_sampler
