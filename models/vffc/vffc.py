import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.nn import conv_nd, conv_nd_class, avg_pool_nd, norm_nd
from models.drop import DropPath


class FFCSEBlock(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSEBlock, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.dims = 3 if ffc3d else 2
        self.conv_layer = conv_nd(dims=self.dims, in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                  out_channels=out_channels * 2, kernel_size=1, stride=1, padding=0, groups=self.groups,
                                  bias=False)

        self.bn = norm_nd(dims=self.dims, num_features=out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)

        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)  # (batch, c, [d], h, w/2+1, 2)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        if self.ffc3d:
            ffted = ffted.permute(0, 1, 5, 2, 3, 4).contiguous()  # (batch, c, 2, d, h, w/2+1)
        else:
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:])
        if self.ffc3d:
            ffted = ffted.permute(0, 1, 3, 4, 5, 2).contiguous()  # (batch,c, d, h, w/2+1, 2)
        else:
            ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, h, w/2+1, 2)

        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, dims, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        self.dims = dims
        ffc3d = dims == 3

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
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, ffc3d=ffc3d, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups, ffc3d=ffc3d)

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


class CrossAttentionBlock(nn.Module):  # 60K params with 128,128,128
    def __init__(self, q_in_channels, kv_in_channels, channels, out_channels, num_heads=1):
        super(CrossAttentionBlock, self).__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.q = nn.Conv1d(q_in_channels, channels, kernel_size=1)
        self.k = nn.Conv1d(kv_in_channels, channels, kernel_size=1)
        self.v = nn.Conv1d(kv_in_channels, channels, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, out_channels, kernel_size=1)
        # todo norm???

    def forward(self, query, key, value):
        """

        :param query: feature map with shape (batch, c, h, w)
        :param key: feature map with shape (batch, c, h, w)
        :param value: feature map with shape (batch, c, h, w)
        :return: feature map with shape (batch, c, h, w)
        """

        b, c_q, *spatial_size = query.shape
        _, c_k, *_ = key.shape
        _, c_v, *_ = value.shape

        query = query.reshape(b, c_q, -1)
        key = key.reshape(b, c_k, -1)
        value = value.reshape(b, c_v, -1)

        q = self.q(query).reshape(b * self.num_heads, -1, query.shape[-1])
        k = self.k(key).reshape(b * self.num_heads, -1, key.shape[-1])
        v = self.v(value).reshape(b * self.num_heads, -1, value.shape[-1])

        scale = 1. / math.sqrt(math.prod(spatial_size))
        attention_weights = torch.bmm(q, k.transpose(1, 2)) * scale
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention = torch.bmm(attention_weights, v)

        attention_output = attention.reshape(b, -1, attention.shape[-1])
        attention_output = self.proj_out(attention_output)
        attention_output = attention_output.reshape(b, -1, *spatial_size)

        return attention_output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, dims, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, use_convolutions=False, cross_attention='none',
                 cross_attention_args=None, dropout_rate=0., **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

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

        if use_convolutions:
            module = nn.Identity if in_cg == 0 or out_cg == 0 else conv_nd_class(dims)
            self.convg2g = module(in_cg, out_cg, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
            self.convg2g = module(in_cg, out_cg, dims, stride, 1 if groups == 1 else groups // 2, enable_lfu,
                                  **spectral_kwargs)

        self.cross_attention = cross_attention
        if cross_attention in ['cross_local', 'cross']:
            self.lg_cross_attention = CrossAttentionBlock(
                q_in_channels=out_cl,
                kv_in_channels=out_cl,
                channels=out_cl // cross_attention_args.get('attention_channel_scale_factor', 1),
                out_channels=out_cl,
                num_heads=cross_attention_args.get('num_heads', 1))
        if cross_attention in ['cross_global', 'cross']:
            self.gl_cross_attention = CrossAttentionBlock(
                q_in_channels=out_cg,
                kv_in_channels=out_cg,
                channels=out_cg // cross_attention_args.get('attention_channel_scale_factor', 1),
                out_channels=out_cg,
                num_heads=cross_attention_args.get('num_heads', 1))

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else conv_nd(dims)
        self.gate = module(in_channels, 2, 1)

        self.dropout_l2l = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout_g2l = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout_g2g = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout_l2g = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

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

            cl2l = self.dropout_l2l(cl2l)
            cg2l = self.dropout_g2l(cg2l)

            if self.cross_attention in ['cross_local', 'cross']:
                out_xl = self.lg_cross_attention(cl2l, cg2l, cg2l)
            else:
                out_xl = cl2l + cg2l
        if self.ratio_gout != 0:
            cl2g = self.convl2g(x_l) * l2g_gate
            cg2g = self.convg2g(x_g)
            cl2g = self.dropout_l2g(cl2g)
            cg2g = self.dropout_g2g(cg2g)

            if self.cross_attention in ['cross_global', 'cross']:
                out_xg = self.gl_cross_attention(cl2g, cg2g, cg2g)
            else:
                out_xg = cl2g + cg2g

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels, dims,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=None, activation_layer=nn.Identity,
                 padding_type='reflect', use_convolutions=False, cross_attention='none',
                 cross_attention_args=None, enable_lfu=True,
                 dropout_rate=0., **kwargs):

        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, dims, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, use_convolutions=use_convolutions,
                       cross_attention=cross_attention, cross_attention_args=cross_attention_args,
                       dropout_rate=dropout_rate, **kwargs)

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
                 spatial_transform_kwargs=None, inline=False, use_convolutions=False,
                 cross_attention='none', cross_attention_args=None, drop_path_rate=0., dropout_rate=0.,
                 **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, dims, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer, padding_type=padding_type,
                                use_convolutions=use_convolutions, cross_attention=cross_attention,
                                cross_attention_args=cross_attention_args,
                                dropout_rate=dropout_rate, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, dims, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer, use_convolutions=use_convolutions,
                                padding_type=padding_type, cross_attention=cross_attention,
                                cross_attention_args=cross_attention_args,
                                dropout_rate=dropout_rate, **conv_kwargs)

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
    def __init__(self, feats_num_bottleneck, norm_layer, n_blocks=9, padding_type='reflect', activation_layer=nn.ReLU,
                 resnet_conv_kwargs={}, spatial_transform_layers=None, spatial_transform_kwargs={},
                 use_convolutions=True, cross_attention='none', cross_attention_args=None, inline=False,
                 dims=2, drop_path_rate=0., dropout_rate=0.):

        super().__init__()

        self.resnet_layers = []
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(dims, feats_num_bottleneck, padding_type=padding_type,
                                          activation_layer=activation_layer,
                                          norm_layer=norm_layer, use_convolutions=use_convolutions,
                                          cross_attention=cross_attention,
                                          cross_attention_args=cross_attention_args,
                                          inline=inline, drop_path_rate=drop_path_rate,
                                          dropout_rate=dropout_rate,
                                          **resnet_conv_kwargs)

            self.resnet_layers += [cur_resblock]

        if not inline:
            self.resnet_layers += [ConcatTupleLayer()]
        self.resnet_layers = nn.Sequential(*self.resnet_layers)

    def forward(self, x):
        x = self.resnet_layers(x)
        return x