import torch
import torch.nn as nn

from models.vffc.vffc import Bottleneck
from models.resnet3d.ResNet3D import generate_model
from models.Decoder2D import Decoder2D
from models.nn import norm_nd_class, normalization

from utils.logger import get_logger

logger = get_logger(__file__)


class VolumetricFFCModel(nn.Module):
    def __init__(self, output_nc=1, drop_path_rate=0.1, input_nc=1):
        super().__init__()

        self.encoder = generate_model(
            model_depth=34,
            n_input_channels=input_nc,
            drop_path_rate=drop_path_rate)

        feats_num_bottleneck = 512
        norm_layer = norm_nd_class(3)
        self.bottleneck = Bottleneck(
            norm_layer=norm_layer,
            feats_num_bottleneck=feats_num_bottleneck,
            inline=True,
            resnet_conv_kwargs={'ratio_gin': 0.75, 'ratio_gout': 0.75},
            drop_path_rate=drop_path_rate)

        self.decoder = Decoder2D(encoder_dims=[64, 128, 256, 512], output_nc=output_nc)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = normalization(x)

        feature_maps = self.encoder(x)
        feature_maps[-1] = self.bottleneck(feature_maps[-1])
        feature_maps = [torch.mean(x, dim=2) for x in feature_maps]

        x = self.decoder(feature_maps)
        return x
