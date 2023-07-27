import numpy as np
import torch
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
import argparse
from pathlib import Path
import sys
import random

from data.papyrus_ink_detection import detect_papyrus_ink
from data.papyrus_loaders import load_papyruses
from data.make_datasets import make_train_dataset, make_eval_dataset, make_test_dataset
from data.samplers import get_samplers

from models import get_model_config, get_model

from utils.model_checkpointing import save_model, load_model, get_last_checkpoint, create_checkpoint_name
from utils.logger import get_logger, DEBUG
from utils.seeding import set_random_seeds, load_and_set_reproducibility_options
from utils.parsing import parse_boolean_args
from utils.dist import setup_dist, get_rank

from training.train import train
from training.losses import get_criterion
from training.optimizers import get_optimizer
from training.schedulers import get_lr_scheduler

from evaluate import evaluate
from transforms.transforms import get_data_load_settings, get_train_transforms, get_eval_transforms

logger = get_logger('main')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    # -------- Parsing arguments --------
    parser = argparse.ArgumentParser(description='vesuvius')

    # -- Transforms --
    parser.add_argument('--enable_train_augmentations', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--dihedral_prob', type=float, default=0.5)
    parser.add_argument('--random_crop_prob', type=float, default=0.5)
    parser.add_argument('--channel_dropout_prob', type=float, default=0.5)
    parser.add_argument('--channel_dropout_max_factor', type=float, default=0.5)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)

    # -- Paths --
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--outputs_path', type=str, required=True)

    # -- Training --
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--z_start', type=int, default=24)
    parser.add_argument('--z_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--train_stride', type=int, default=64)
    parser.add_argument('--eval_stride', type=int, default=64)
    parser.add_argument('--test_stride', type=int, default=64)
    parser.add_argument('--test_every_epochs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size_train', type=int, default=4)
    parser.add_argument('--batch_size_eval', type=int, default=4)  
    parser.add_argument('--resume', type=str, default='none')

    # -- Optimization --
    parser.add_argument('--lr_scheduler', type=str, default='one_cycle', choices=['one_cycle', 'constant'])
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--loss', type=str, default='bce_dice', choices=['bce', 'dice', 'bce_dice'])
    parser.add_argument('--weigh_bce', type=float, default=1)
    parser.add_argument('--optimizer', type=str, default='adamW', choices=['sgd', 'adamW'])

    # -- Performance --
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--cudnn_deterministic', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--use_amp', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--amp_dtype', type=str, default='float16', choices=['float16', 'none'])
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--use_zero_optim', type=str, default='False', choices=['True', 'False'])

    args = parser.parse_args()

    args = parse_boolean_args(args)
    args = setup_dist(args)

    args.resume = None if args.resume == 'none' else args.resume
    args.checkpoint_path = Path(args.outputs_path) / 'vesuvius_checkpoints'
    args.amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
    args.z_start = args.z_start if args.z_size != 65 else 0

    if args.z_size < 16:
        logger.warning('z_size should be at least 16')
    if args.enable_train_augmentations:
        assert 16 + args.z_size <= 65, '2 * z_swing + z_size should be at most 65'

    args.model_config = get_model_config(args)
    model = get_model(args.model_config).to(args.device)

    if args.resume:
        last_checkpoint, args.experiment_name = get_last_checkpoint(args.checkpoint_path, args.resume)
        g = load_and_set_reproducibility_options(last_checkpoint)
    else:
        args.experiment_name = create_checkpoint_name(args)
        g = set_random_seeds(args.seed, args.cudnn_deterministic)
        g.manual_seed(args.seed)

    logger.info(f'Experiment {args.experiment_name}. Running on {args.node}. {str("Resumed") if args.resume else ""}')
    args.outputs_path = Path(args.outputs_path) / 'vesuvius_detections' / f'{args.experiment_name}'
    if not args.outputs_path.exists():
        args.outputs_path.mkdir(parents=True)

    args.transforms_settings, args.load_settings = get_data_load_settings(
        args=args,
        load_patch_size=args.patch_size,
        load_z_start=args.z_start,
        load_z_size=args.z_size)

    papyruses_train_paths = sorted(list(Path(args.train_data_path).iterdir()))
    papyruses = load_papyruses(
        train_data_paths=papyruses_train_paths,
        load_patch_size=args.load_settings['load_patch_size'],
        load_z_start=args.load_settings['load_z_start'],
        load_z_size=args.load_settings['load_z_size'],
        patch_size=args.patch_size,
        z_start=args.z_start,
        z_size=args.z_size,
        load_labels=True)

    train_transforms = get_train_transforms(args.transforms_settings, args.patch_size, args.z_size)
    train_dataset = make_train_dataset(
        papyruses=papyruses,
        patch_size=args.patch_size,
        load_patch_size=args.load_settings['load_patch_size'],
        load_z_start=args.load_settings['load_z_start'],
        load_z_size=args.load_settings['load_z_size'],
        z_start=args.z_start,
        z_size=args.z_size,
        train_stride=args.train_stride,
        train_transforms=train_transforms
    )

    eval_transforms = get_eval_transforms(args.transforms_settings, args.z_size)
    eval_dataset = make_eval_dataset(
        papyruses=papyruses,
        patch_size=args.patch_size,
        z_start=args.z_start,
        z_size=args.z_size,
        eval_stride=args.eval_stride,
        eval_transforms=eval_transforms
    )

    test_dataset = make_test_dataset(
        papyruses=papyruses,
        patch_size=args.patch_size,
        z_start=args.z_start,
        z_size=args.z_size,
        test_stride=args.test_stride,
        test_transforms=eval_transforms
    )

    train_sampler, eval_sampler = get_samplers(train_dataset, eval_dataset, args.seed, g)

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=int(args.batch_size_train // args.world_size),
        sampler=train_sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
        pin_memory=args.pin_memory
    )

    eval_loader = data.DataLoader(
        dataset=eval_dataset,
        batch_size=int(args.batch_size_eval // args.world_size),
        sampler=eval_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory)

    criterion = get_criterion(loss=args.loss, weigh_bce=args.weigh_bce, device=args.device)
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    lr_scheduler = get_lr_scheduler(args.lr_scheduler, args.learning_rate, optimizer, len(train_loader), args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    start_epoch = 0
    best_eval_fbeta = 0.0
    if args.resume:
        model, optimizer, lr_scheduler, scaler, training_state, training_config = load_model(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            save_path=last_checkpoint)

        start_epoch = training_state['epoch']
        best_eval_fbeta = training_state['best_eval_fbeta']
        args.exp_name = training_config['exp_name']
        args.epochs = training_config['epochs']

    model_saver = save_model(args.experiment_name, args.epochs, args.seed, args.cudnn_deterministic,
                             args.checkpoint_path, vars(args))

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    if args.use_zero_optim:
        optimizer = ZeroRedundancyOptimizer(model.parameters(), type(optimizer), lr=args.learning_rate)
        logger.info(f"Using ZeroRedundancyOptimizer")

    logger.info("Training...")
    for epoch in range(start_epoch, args.epochs):

        train_epoch_loss, train_epoch_fbeta, start_epoch_time = train(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            scaler=scaler,
            train_loader=train_loader,
            epochs=args.epochs,
            device=args.device,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
            epoch=epoch
        )

        logger.info(f'Epoch [{epoch}]/[{args.epochs}]. Train Loss: {train_epoch_loss}. '
                    f'Train Fbeta: {train_epoch_fbeta}. Evaluation...')

        eval_epoch_loss, eval_epoch_fbeta = evaluate(
            model=model,
            criterion=criterion,
            eval_loader=eval_loader,
            epochs=args.epochs,
            device=args.device,
            epoch=epoch
        )

        if args.distributed:
            dist.all_reduce(eval_epoch_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(eval_epoch_fbeta, op=dist.ReduceOp.SUM)
            eval_epoch_loss = (eval_epoch_loss / args.world_size)
            eval_epoch_fbeta = (eval_epoch_fbeta / args.world_size)

        eval_epoch_loss = eval_epoch_loss.item()
        eval_epoch_fbeta = eval_epoch_fbeta.item()

        logger.info(f'Epoch [{epoch}]/[{args.epochs}]. Eval Loss: {eval_epoch_loss}. '
                    f'Eval Fbeta: {eval_epoch_fbeta}. Evaluation...')

        if eval_epoch_fbeta > best_eval_fbeta:
            best_eval_fbeta = eval_epoch_fbeta
            logger.info(f'New best eval fbeta: {best_eval_fbeta}')

            if args.use_zero_optim:
                optimizer.consolidate_state_dict()

            model_saver(
                model=model.module if args.distributed else model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                best_eval_fbeta=best_eval_fbeta,
                epoch=epoch,
                generator_state=train_loader.generator.get_state(),
                scaler=scaler,
                filename=f'{args.experiment_name}_{best_eval_fbeta:.02f}')

        if epoch % args.test_every_epochs == args.test_every_epochs - 1 and get_rank() == 0:
            logger.info(f"Epoch [{epoch}]/[{args.epochs}]. Starting test...")

            result = detect_papyrus_ink(
                model=model,
                dataset=test_dataset,
                batch_size=args.batch_size_eval,
                threshold=args.threshold,
                load_labels=True,
                device=args.device,
                epoch=epoch,
                save_path=args.outputs_path,
            )

        if args.distributed:
            torch.distributed.barrier()

        if args.use_zero_optim:
            optimizer.consolidate_state_dict()

        model_saver(
            model=model.module if args.distributed else model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            best_eval_fbeta=best_eval_fbeta,
            epoch=epoch,
            generator_state=train_loader.generator.get_state(),
            scaler=scaler,
            filename=f'{args.experiment_name}')

    if args.distributed:
        torch.distributed.barrier()

    sys.exit()


if __name__ == '__main__':
    main()
