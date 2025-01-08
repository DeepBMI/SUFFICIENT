import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
# from labml_helpers.module import Module
from torch.nn import Module

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module


def conv(in_f, out_f, kernel_size, stride=1, pad='zero'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)

def decodernw(
        num_output_channels=1,
        num_channels_up=[128 ] *5,
        filter_size_up=1,
        need_sigmoid=True,
        pad ='reflection',
        upsample_mode='bilinear',
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True)
        bn_before_act = False,
        bn_affine = True,
        upsample_first = True,
):

    num_channels_up = num_channels_up + [num_channels_up[-1] ,num_channels_up[-1]]
    n_scales = len(num_channels_up)

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up ] *n_scales
    model = nn.Sequential()


    for i in range(len(num_channels_up ) -1):

        if upsample_first:
            model.add(conv( num_channels_up[i], num_channels_up[ i +1],  filter_size_up[i], 1, pad=pad))
            if upsample_mode !='none' and i != len(num_channels_up ) -2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            # model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))
        else:
            if upsample_mode !='none' and i!= 0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            # model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))
            model.add(conv(num_channels_up[i], num_channels_up[i + 1], filter_size_up[i], 1, pad=pad))

        if i != len(num_channels_up) - 1:
            if (bn_before_act):
                model.add(nn.BatchNorm2d(num_channels_up[i + 1], affine=bn_affine))
            model.add(act_fun)
            if (not bn_before_act):
                model.add(nn.BatchNorm2d(num_channels_up[i + 1], affine=bn_affine))

    model.add(conv(num_channels_up[-1], num_output_channels, 1, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


class Deep_Decoder(Module):
    def __init__(self, num_output_channels=3,num_channels_up= [128] *5,
        filter_size_up=1,need_sigmoid=True,need_relu=False,pad ='reflection',upsample_mode='trilinear',
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True)
        bn_before_act = False,bn_affine = True,upsample_first = True,):
        super().__init__()

        num_channels_up = num_channels_up + [num_channels_up[-1], num_channels_up[-1]]
        # num_channels_up = num_channels_up + [32, 16]
        # num_channels_up = num_channels_up + [8, 8]
        # num_channels_up = num_channels_up + [128, 64]
        n_scales = len(num_channels_up)

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
            filter_size_up = [filter_size_up] * n_scales
        decoder = nn.Sequential()

        for i in range(len(num_channels_up) - 1):

            if upsample_first:
                decoder.add(self.conv(num_channels_up[i], num_channels_up[i + 1], filter_size_up[i], 1, pad=pad))
                if upsample_mode != 'none' and i != len(num_channels_up) - 2:
                    if i > 1:
                        decoder.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
                # model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))
            else:
                if upsample_mode != 'none' and i != 0:
                    decoder.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
                # model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))
                decoder.add(self.conv(num_channels_up[i], num_channels_up[i + 1], filter_size_up[i], 1, pad=pad))

            if i != len(num_channels_up) - 1:
                if (bn_before_act):
                    decoder.add(nn.BatchNorm3d(num_channels_up[i + 1]))
                decoder.add(act_fun)
                if (not bn_before_act):
                    decoder.add(nn.BatchNorm3d(num_channels_up[i + 1]))

        decoder.add(self.conv(num_channels_up[-1], num_channels_up[-1], 1, pad=pad))



        if need_sigmoid:
            decoder.add(nn.Sigmoid())
            # decoder.add(nn.Softplus())
        if need_relu:
            decoder.add(nn.ReLU())
        decoder.add(nn.BatchNorm3d(num_channels_up[-1]))
        decoder.add(self.conv(num_channels_up[-1], num_output_channels, 1, pad=pad))
        self.decoder = decoder

    def conv(self,in_f, out_f, kernel_size, stride=1, pad='zero'):
        padder = None
        to_pad = int((kernel_size - 1) / 2)
        if pad == 'reflection':
            padder = nn.ReflectionPad3d(to_pad)
            to_pad = 0

        convolver = nn.Conv3d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

        layers = filter(lambda x: x is not None, [padder, convolver])
        return nn.Sequential(*layers)

    def forward(self,x):
        return self.decoder(x)



class Deep_Decoder_Body(Module):
    def __init__(self, num_output_channels=3,num_channels_up= [128] *5,
        filter_size_up=1,need_sigmoid=True,need_relu=False,pad ='reflection',upsample_mode='trilinear',
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True)
        bn_before_act = False,bn_affine = True,upsample_first = True,):
        super().__init__()

        # num_channels_up = num_channels_up + [num_channels_up[-1], num_channels_up[-1]]
        num_channels_up = num_channels_up + [16, 8]
        n_scales = len(num_channels_up)

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
            filter_size_up = [filter_size_up] * n_scales
        decoder = nn.Sequential()

        for i in range(len(num_channels_up) - 1):

            if upsample_first:
                decoder.add(self.conv(num_channels_up[i], num_channels_up[i + 1], filter_size_up[i], 1, pad=pad))
                if upsample_mode != 'none' and i != len(num_channels_up) - 2:
                    decoder.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
                # model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))
            else:
                if upsample_mode != 'none' and i != 0:
                    decoder.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
                # model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))
                decoder.add(self.conv(num_channels_up[i], num_channels_up[i + 1], filter_size_up[i], 1, pad=pad))

            if i != len(num_channels_up) - 1:
                if (bn_before_act):
                    decoder.add(nn.BatchNorm3d(num_channels_up[i + 1], affine=bn_affine))
                decoder.add(act_fun)
                if (not bn_before_act):
                    decoder.add(nn.BatchNorm3d(num_channels_up[i + 1], affine=bn_affine))

        decoder.add(self.conv(num_channels_up[-1], num_output_channels, 1, pad=pad))
        if need_sigmoid:
            # decoder.add(nn.Sigmoid())
            decoder.add(nn.Softplus())
        if need_relu:
            decoder.add(nn.ReLU())
        self.decoder = decoder

    def conv(self,in_f, out_f, kernel_size, stride=1, pad='zero'):
        padder = None
        to_pad = int((kernel_size - 1) / 2)
        if pad == 'reflection':
            padder = nn.ReflectionPad3d(to_pad)
            to_pad = 0

        convolver = nn.Conv3d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

        layers = filter(lambda x: x is not None, [padder, convolver])
        return nn.Sequential(*layers)

    def forward(self,x):
        return self.decoder(x)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_f, out_f):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_f, out_f, 1, 1, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return out


def resdecoder(
        num_output_channels=3,
        num_channels_up=[128] * 5,
        filter_size_up=1,
        need_sigmoid=True,
        pad='reflection',
        upsample_mode='bilinear',
        act_fun=nn.ReLU(),  # nn.LeakyReLU(0.2, inplace=True)
        bn_before_act=False,
        bn_affine=True,
):
    num_channels_up = num_channels_up + [num_channels_up[-1], num_channels_up[-1]]
    n_scales = len(num_channels_up)

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    model = nn.Sequential()

    for i in range(len(num_channels_up) - 2):

        model.add(ResidualBlock(num_channels_up[i], num_channels_up[i + 1]))

        if upsample_mode != 'none':
            model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            # model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))

        if i != len(num_channels_up) - 1:
            model.add(act_fun)
            # model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))

    # new
    model.add(ResidualBlock(num_channels_up[-1], num_channels_up[-1]))
    # model.add(nn.BatchNorm2d( num_channels_up[-1] ,affine=bn_affine))
    model.add(act_fun)
    # end new

    model.add(conv(num_channels_up[-1], num_output_channels, 1, pad=pad))

    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


class CNNEncoder(nn.Module):

    class Encoder(nn.Module):

        def __init__(self, in_channel, out_channel, norm=nn.InstanceNorm3d, use_skip=False, bias=False):
            super(CNNEncoder.Encoder, self).__init__()

            self._input = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1, bias=bias),
                norm(out_channel),
                nn.ReLU()
            )
            self._output = nn.Sequential(
                nn.Conv3d(out_channel, out_channel, 3, 1, 1, bias=bias),
                norm(out_channel),
                nn.ReLU()
            )

            self.use_skip = use_skip

        def forward(self, feature_map):

            mid = self._input(feature_map)

            res = self._output(mid) if not self.use_skip else mid + self._output(mid)

            return res

    def __init__(self, depth, base, norm):
        super(CNNEncoder, self).__init__()

        self.__input = CNNEncoder.Encoder(1, base, norm=norm)

        self.__encoders = nn.ModuleList([nn.Sequential(nn.MaxPool3d(2),
                                                       CNNEncoder.Encoder(base * 2 ** i, base * 2 ** (i + 1), norm))
                                         for i in range(depth)])

    def forward(self, x):
        skips = []

        in_encoder = self.__input(x)

        skips.append(in_encoder)

        for encoder in self.__encoders:
            in_encoder = encoder(in_encoder)

            skips.append(in_encoder)

        return in_encoder, skips


class CNNDecoder(nn.Module):

    class Decoder(nn.Module):

        def __init__(self, in_channel, out_channel, block_num, norm=nn.InstanceNorm3d, use_skip=False, bias=True):
            super(CNNDecoder.Decoder, self).__init__()

            self.__input = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1, bias=bias),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

            self.__mid = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1, bias=bias),
                norm(out_channel),
                nn.ReLU()
            )

            self._output = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(out_channel, out_channel, 3, 1, 1, bias=bias),
                    norm(out_channel),
                    nn.ReLU()
                )
                for _ in range(block_num)
            ])

            self.use_skip = use_skip

        def forward(self, x, skip):
            x = self.__input(x)

            mid = self.__mid(torch.cat([x, skip], dim=1))

            return self._output(mid) if self.use_skip else mid + self._output(mid)

    def __init__(self, depth, base, block_num, norm, use_skip):
        super(CNNDecoder, self).__init__()

        self.__decoders = nn.ModuleList(
            [CNNDecoder.Decoder(base * 2 ** i, base * 2 ** (i - 1), block_num, norm, use_skip)
             for i in range(depth, 0, -1)])

    def forward(self, x, skips):
        skips.pop()

        skips.reverse()

        for decoder, skip in zip(self.__decoders, skips):
            x = decoder(x, skip)

        return x


class DIPNet(nn.Module):

    def __init__(self, depth, base, decoder_block_num, norm=nn.InstanceNorm3d, encoder_norm=nn.Identity, use_skip=False):
        super(DIPNet, self).__init__()

        self.encoder = CNNEncoder(depth, base, encoder_norm)

        self.decoder = CNNDecoder(depth, base, decoder_block_num, norm=norm, use_skip=use_skip)

        self.output = nn.Conv3d(base, 1, 1, 1, 0, bias=False)

    def forward(self, x):

        btm, skips = self.encoder(x)

        top = self.decoder(btm, skips)

        return self.output(top)