import random
import numpy as np
import torch
import os


def set_random_seeds(seed, cudnn_deterministic=True):
    """
    Set the random seeds. Call for each process.
    https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    """
    random.seed(seed)  # Python random number generator
    np.random.seed(seed)  # Numpy random number generator
    torch.manual_seed(seed)  # PyTorch random number generator
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.use_deterministic_algorithms(cudnn_deterministic)
        torch.backends.cudnn.benchmark = not cudnn_deterministic
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    return g


def load_and_set_reproducibility_options(path):
    checkpoint = torch.load(path)
    random_settings = checkpoint['random_states']
    seed = random_settings['seed']

    if isinstance(random_settings['cudnn_deterministic'], bool):
        cudnn_deterministic = random_settings['cudnn_deterministic']
    else:
        cudnn_deterministic = random_settings['cudnn_deterministic'] == 'True'
    generator_state = random_settings['generator_state']
    set_random_seeds(seed, cudnn_deterministic)

    g = torch.Generator()
    g.set_state(generator_state)
    random.setstate(random_settings['random_rng_state'])
    np.random.set_state(random_settings['numpy_rng_state'])
    torch.set_rng_state(random_settings['torch_rng_state'])
    torch.cuda.set_rng_state(random_settings['cuda_rng_state'])
    torch.cuda.empty_cache()

    return g
