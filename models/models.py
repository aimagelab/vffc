from models.vffc_model import VolumetricFFCModel


def get_model_config(args):

    model_config = {'drop_path_rate': args.drop_path_rate}

    return model_config


def get_model(model_config: dict):
    model = VolumetricFFCModel(**model_config)
    return model
