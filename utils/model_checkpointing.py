from pathlib import Path
import numpy as np
import random
import shutil
import torch
import uuid

from utils.dist import get_rank


def save_model(exp_name, epochs, seed, cudnn_deterministic, save_path, config_dict):
    save_path = Path(save_path) / exp_name
    if not save_path.exists():
        save_path.mkdir(parents=True)

    def save(model, optimizer, lr_scheduler, best_eval_fbeta, epoch, generator_state, scaler, filename):

        if get_rank() != 0:
            return

        checkpoint_path = Path(save_path) / f'{filename}.pth'
        random_states = {'random_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state(), 'seed': seed,
                         'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state(),
                         'cudnn_deterministic': cudnn_deterministic, 'generator_state': generator_state}

        state = {
            'exp_name': exp_name,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_eval_fbeta': best_eval_fbeta,
            'epoch': epoch,
            'epochs': epochs,
            'scaler': scaler.state_dict(),
            'random_states': random_states,
            'config_dict': config_dict
        }
        torch.save(state, checkpoint_path)

        # Keep a last version copy of the model saved for easier training resume
        shutil.copyfile(checkpoint_path, save_path / f'{exp_name}_last.pth')

    return save


def create_checkpoint_name(args):

    exp_name = [
        str(uuid.uuid4())[:4],
        f'PS_{args.patch_size}',
        f'ZSi_{args.z_size}',
        f'ZSt_{args.z_start}',
        f'BS_{args.batch_size_train}',
        f'L_{args.loss}',
        f'BCE_W_{int(args.weigh_bce)}' if args.weigh_bce != 1 else None,
        'DIH' if args.dihedral_prob > 0 else None,
        'RC' if args.random_crop_prob > 0 else None,
        f'CH_DR_{args.channel_dropout_max_factor}' if args.channel_dropout_prob > 0 else None,
    ]
    experiment_name = '-'.join(n for n in exp_name if n is not None)
    return experiment_name


def get_last_checkpoint(checkpoint_path, model_uuid):
    checkpoint_dir = list(Path(checkpoint_path).glob(f'{model_uuid}*'))[0]
    last_checkpoint = list(checkpoint_dir.glob('*last.pth'))
    assert len(last_checkpoint) == 1, f'Found {len(last_checkpoint)} checkpoints with uuid {model_uuid}'
    last_checkpoint = last_checkpoint[0]
    experiment_name = last_checkpoint.stem.rstrip('_last')
    return last_checkpoint, experiment_name


def load_model(model, optimizer, lr_scheduler, scaler, save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])

    training_state = {
        'epoch': checkpoint['epoch'] + 1,
        'best_eval_fbeta': checkpoint['best_eval_fbeta'],
    }

    training_config = {
        'exp_name': checkpoint['exp_name'],
        'epochs': checkpoint['epochs'],
    }

    return model, optimizer, lr_scheduler, scaler, training_state, training_config
