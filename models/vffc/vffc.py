import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.nn import conv_nd, conv_nd_class, avg_pool_nd, norm_nd
from models.drop import DropPath


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.dims = 3
        self.conv_layer = conv_nd(dims=self.dims, in_channels=in_channels * 2,
                                  out_channels=out_channels * 2, kernel_size=1, stride=1, padding=0, groups=self.groups,
                                  bias=False)

        self.bn = norm_nd(dims=self.dims, num_features=out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.ffc3d = True
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        ffted = torch.fft.rfftn(x, dim=(-3, -2, -1), norm=self.fft_norm)  # (batch, c, [d], h, w/2+1, 2)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 5, 2, 3, 4).contiguous()  # (batch, c, 2, d, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:])
        ffted = ffted.permute(0, 1, 3, 4, 5, 2).contiguous()  # (batch,c, d, h, w/2+1, 2)

        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=(-3, -2, -1), norm=self.fft_norm)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, dims, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        self.dims = dims

        if stride == 2:
            self.downsample = avg_pool_nd(dims=dims, kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            conv_nd(dims=dims, in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1,
                    groups=groups, bias=False),
            norm_nd(dims=dims, num_features=out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)

        self.conv2 = conv_nd(dims=dims, in_channels=out_channels // 2, out_channels=out_channels, kernel_size=1,
                             groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            if self.dims == 3:
                n, c, d, h, w = x.shape
            else:
                n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            if self.dims == 3:
                xs = xs.repeat(1, 1, 1, split_no, split_no).contiguous()
            else:
                xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, dims, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        convnd_kwargs = dict(
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_type
        )

        module = nn.Identity if in_cl == 0 or out_cl == 0 else conv_nd_class(dims)
        self.convl2l = module(in_cl, out_cl, **convnd_kwargs)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else conv_nd_class(dims)
        self.convl2g = module(in_cl, out_cg, **convnd_kwargs)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else conv_nd_class(dims)
        self.convg2l = module(in_cg, out_cl, **convnd_kwargs)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, dims, stride, 1 if groups == 1 else groups // 2, enable_lfu,
                              **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else conv_nd(dims)
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            cl2l = self.convl2l(x_l)
            cg2l = self.convg2l(x_g) * g2l_gate

            out_xl = cl2l + cg2l

        if self.ratio_gout != 0:
            cl2g = self.convl2g(x_l) * l2g_gate
            cg2g = self.convg2g(x_g)

            out_xg = cl2g + cg2g

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels, dims,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=None, activation_layer=nn.Identity,
                 padding_type='reflect', enable_lfu=True, **kwargs):

        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, dims, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)

        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dims, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 inline=False, drop_path_rate=0., **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, dims, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer, padding_type=padding_type, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, dims, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)

        self.inline = inline
        self.drop_path_local = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.drop_path_global = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l = id_l + self.drop_path_local(x_l)
        x_g = id_g + self.drop_path_global(x_g)

        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class Bottleneck(nn.Module):
    def __init__(self, feats_num_bottleneck, norm_layer, n_blocks=3, padding_type='reflect', activation_layer=nn.ReLU,
                 resnet_conv_kwargs={}, inline=False, drop_path_rate=0.):

        super().__init__()

        self.resnet_layers = []
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(3, feats_num_bottleneck, padding_type=padding_type,
                                          activation_layer=activation_layer,
                                          norm_layer=norm_layer, inline=inline, drop_path_rate=drop_path_rate,
                                          **resnet_conv_kwargs)

            self.resnet_layers += [cur_resblock]

        if not inline:
            self.resnet_layers += [ConcatTupleLayer()]
        self.resnet_layers = nn.Sequential(*self.resnet_layers)

    def forward(self, x):
        x = self.resnet_layers(x)
        return x