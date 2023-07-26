
def parse_boolean_args(args):
    args.use_amp = args.use_amp == 'True'
    args.use_zero_optim = args.use_zero_optim == 'True'
    args.pin_memory = args.pin_memory == 'True'
    args.enable_train_augmentations = args.enable_train_augmentations == 'True'
    args.cudnn_deterministic = args.cudnn_deterministic == 'True'

    return args
