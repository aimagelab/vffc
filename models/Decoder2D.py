import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder2D(nn.Module):
    def __init__(self, encoder_dims, output_nc=1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        last_channel_size = encoder_dims[0]
        self.logit = nn.Conv2d(last_channel_size, output_nc, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = feature_maps[0]

        x = self.logit(x)
        mask = self.up(x)
        return mask
