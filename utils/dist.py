import torch
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
import os

from utils.logger import get_logger

logger = get_logger(__file__)


def setup_dist(args):
    args.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    args.local_rank = 0
    args.rank = 0
    args.node = os.uname()[1]
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = 1
    args.n_gpus_per_node = torch.cuda.device_count()
    if args.distributed:
        args.device = torch.device('cuda', args.local_rank)
        torch.cuda.set_device(args.device)
        args.world_size = int(os.environ["WORLD_SIZE"])

        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', rank=args.rank, world_size=args.world_size)

        logger.info(f'Train in distributed mode with multiple processes, 1 GPU/process. Process: '
                    f'{args.local_rank}, global rank: [{args.rank}/{args.world_size}], device: {args.device}, '
                    f'GPUs per node: {args.n_gpus_per_node}. Node: {args.node}')

        torch.distributed.barrier()
    else:
        logger.info(f'Training with a single process on 1 GPUs, device: {args.device}')
    assert args.local_rank >= 0
    return args


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return int(os.environ['RANK'])


def get_local_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return int(os.environ['WORLD_SIZE'])
