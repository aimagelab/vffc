import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.drop import DropPath
from models.nn import conv_nd, norm_nd, max_pool_nd

in_planes = [64, 128, 256, 512]


def conv3x3x3_nd(dims, in_planes, out_planes, stride=1):
    return conv_nd(dims, in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1_nd(dims, in_planes, out_planes, stride=1):
    return conv_nd(dims, in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, dims, in_planes, planes, stride=1, downsample=None, drop_path_rate=0.0):
        super().__init__()

        if isinstance(stride, tuple):
            stride = stride if dims == 3 else stride[1:]
        self.conv1 = conv3x3x3_nd(dims, in_planes, planes, stride)
        self.bn1 = norm_nd(dims, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3_nd(dims, planes, planes)
        self.bn2 = norm_nd(dims, planes)
        self.downsample = downsample
        self.stride = stride

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + self.drop_path(out)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 shortcut_type='B',
                 widen_factor=1.0,
                 drop_path_rate=0.0,
                 dims=3):

        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.dims = dims

        self.conv1 = conv_nd(
            dims=3,
            in_channels=n_input_channels,
            out_channels=self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False)

        self.bn1 = norm_nd(3, self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = max_pool_nd(
            dims=3,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )

        self.layer1 = self._make_layer(block,
                                       block_inplanes[0],
                                       layers[0],
                                       shortcut_type,
                                       stride=(1, 1, 1),
                                       downsample=False,
                                       drop_path_rate=drop_path_rate)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=(1, 2, 2),
                                       downsample=True,
                                       drop_path_rate=drop_path_rate)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=(1, 2, 2),
                                       downsample=True,
                                       drop_path_rate=drop_path_rate)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=(1, 2, 2),
                                       downsample=True,
                                       drop_path_rate=drop_path_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        if self.dims == 2:
            out = F.avg_pool2d(x, kernel_size=1, stride=stride[1:])
            zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                    out.size(3), out.size(4))
        else:
            out = F.avg_pool3d(x, kernel_size=1, stride=stride)
            zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                    out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride, downsample, drop_path_rate):
        downsample_block = None
        if downsample:
            if shortcut_type == 'A':
                downsample_block = partial(self._downsample_basic_block,
                                           planes=planes * block.expansion,
                                           stride=stride)
            else:
                downsample_block = nn.Sequential(
                    conv1x1x1_nd(3, self.in_planes, planes * block.expansion, stride),
                    norm_nd(3, planes * block.expansion))

        layers = []
        layers.append(
            block(dims=3,
                  in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample_block,
                  drop_path_rate=drop_path_rate))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(3, self.in_planes, planes, drop_path_rate=drop_path_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


def generate_model(model_depth, **kwargs):
    assert model_depth in [34]

    if model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], in_planes, **kwargs)
    else:
        raise ValueError('Unsupported model depth, must one 34')

    return model
