import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv.ops as MOP
import Modules.base as B
from torch.nn.modules.utils import _pair, _quadruple
from torchvision.transforms.functional import gaussian_blur
from util import ImagePadder

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, bias=False, activation="relu", norm=None):
        super(ConvLayer, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.act = getattr(torch, str(activation), nn.Identity())

        if norm == "BN":
            self.norm = nn.BatchNorm2d(out_channels, track_running_stats=True)
        elif norm == "IN":
            self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
class InvBottle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, exp=4, activation=None, norm=None):

        super(InvBottle, self).__init__()
        mid_channels = in_channels * exp
        # D = 1
        # self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=groups, bias=False)
        if kernel_size > 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * exp, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=in_channels, bias=False),
                nn.Conv2d(in_channels * exp, out_channels, kernel_size=1, groups=groups, bias=bias),)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, stride, padding, bias=bias)
        
        
        self.act = getattr(torch, str(activation), nn.Identity())
 
        if norm == "BN":
            self.norm = nn.BatchNorm2d(out_channels, track_running_stats=True)
        elif norm == "IN":
            self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        else:
            self.norm = nn.Identity()

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if bias:
            if kernel_size > 1:
                nn.init.zeros_(self.conv[1].bias)
            else:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # self.conv1.weight.data = self.conv1.weight * self.mask

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# class InvBottle(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
#                  padding=0, dilation=1, groups=1, bias=False, exp=4, activation=None, norm=None):

#         super(InvBottle, self).__init__()
#         mid_channels = in_channels * exp
#         # D = 1
#         if kernel_size > 1:
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels * exp, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=in_channels, bias=False),
#                 nn.Conv2d(in_channels * exp, out_channels, kernel_size=1, groups=groups, bias=bias),)
#         else:
#             self.conv = nn.Conv2d(in_channels, out_channels, 1, stride, padding, bias=bias)
        
#         self.act = getattr(torch, str(activation), nn.Identity())
 
#         if norm == "BN":
#             self.norm = nn.BatchNorm2d(out_channels, track_running_stats=True)
#         elif norm == "IN":
#             self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=True)
#         else:
#             self.norm = nn.Identity()

#         self.out_channels = out_channels
#         self.kernel_size = kernel_size

#     def forward(self, x):
#         # self.conv1.weight.data = self.conv1.weight * self.masks
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.act(x)
#         return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, dilation=dilation, groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(out_channels * 5, out_channels, 1, groups=out_channels, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        pad = [1, 1, 1, 1]
        x = F.pad(x, pad)
        # if x.size(2) % 2 == 1:
        #     pad[2] += 1
        # if x.size(3) % 2 == 1:
        #     pad[0] += 1
        xc = self.conv1(x[..., 1:-1, 1:-1])
        xt = self.conv1(x[..., 0:-2, 1:-1])
        xb = self.conv1(x[..., 2:,   1:-1])
        xl = self.conv1(x[..., 1:-1, 0:-2])
        xr = self.conv1(x[..., 1:-1, 2:  ])

        x = torch.cat([xc, xt, xb, xl, xr], dim=1)
        x = self.conv2(x)
        x = self.act(x)
        # x = self.avgpool(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x

class ResidualBlock(nn.Module):
    """
    Residual block as in "Deep residual learning for image recognition", He et al. 2016.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(self, in_channels, out_channels,
                stride=1, activation="relu", norm=None, **kwargs):
        super(ResidualBlock, self).__init__()
        bias = False if norm == "BN" else True
        self.conv1 = InvBottle(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            activation=activation,
            norm=norm
        )
        # self.conv2 = InvBottle(
        #     out_channels,
        #     out_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=bias,
        #     activation=activation,
        #     norm=norm
        # )
        self.act = getattr(torch, str(activation), nn.Identity())

        # self.squeeze = nn.Sequential(
        #     nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1, padding=0, bias=False),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, padding=0, bias=False),
        #     nn.Sigmoid()
        # )
        self.squeeze = nn.Sequential(
            ChannelAttention(in_channels, ratio=4),
            SpatialAttention()
        )

    def forward(self, x):
        out = self.conv1(x)
        # x = self.conv2(x)
        
        # x = self.act(x)
        # x = x + residual
        
        out = self.act(out)
        out = self.squeeze(out)
        out += x
        # out = self.act(out)
        return out

class ConvGRU(nn.Module):
    """
    Convolutional GRU cell.
    Adapted from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation='tanh', conv_func=nn.Conv2d):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_size = hidden_size
        self.reset_gate = conv_func(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = conv_func(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = conv_func(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        # assert activation is None, "ConvGRU activation cannot be set (just for compatibility)"
        self.act = getattr(torch, str(activation), nn.Identity())

    def forward(self, x, state):
        # generate empty prev_state, if None is provided
        if state is None:
            size = (x.size(0), self.hidden_size,) +  x.size()[2:]
            state = torch.zeros(size).to(x)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([x, state], dim=1)
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        x = self.act(self.out_gate(torch.cat([x, state * reset], dim=1)))
        # x = torch.tanh(self.out_gate(torch.cat([x, state * reset], dim=1)))

        state = state * (1 - update) + x * update

        return state, state

class ConvLSTM(nn.Module):
    """
    Convolutional LSTM module.
    Adapted from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation, conv_func=nn.Conv2d):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = conv_func(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)
        self.act = getattr(torch, str(activation), nn.Identity())

    def forward(self, x, prev_state=None):

        # generate empty prev_state, if None is provided
        if prev_state is None:
            # create the zero tensor if it has not been created already
            size = tuple([x.size(0), self.hidden_size] + list(x.size()[2:]))
            if size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[size] = (
                    torch.zeros(size).to(x),
                    torch.zeros(size).to(x),
                )

            prev_state = self.zero_tensors[size]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((x, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = self.act(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * self.act(cell)

        return hidden, (hidden, cell)

# class Decouple(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=1, dilation=1, bias=False):

#         super(Decouple, self).__init__()
#         assert kernel_size > 1, "Not support 1 * 1 Conv"

#         D = kernel_size * 2 - 2
#         # self.conv1 = nn.Conv2d(in_channels, in_channels * D, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=in_channels, bias=False)
#         self.kernel_size = kernel_size
#         # self.conv_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,  kernel_size=3, padding=1, stride=stride, dilation=dilation, bias=False)
#         # self.conv_offset = nn.Conv2d(in_channels, 4 * 2 * kernel_size * kernel_size,  kernel_size=3, padding=1, stride=stride, dilation=dilation, bias=True)
#         self.conv_offset = InvBottle(in_channels, 4 * 2 * kernel_size * kernel_size,  kernel_size=3, padding=1, stride=stride, dilation=dilation, bias=True, diff=True)

#         self.conv_m = nn.ModuleDict(
#             {
#                 'conv1': MOP.DeformConv2d(in_channels, in_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, groups=in_channels, deform_groups=4),
#                 'conv2': nn.Conv2d(in_channels, out_channels, 1, bias=False), 
#             }
#         )
        
#     # def init_weights(self):
#     #     super().init_weights()
#     #     if hasattr(self, 'conv_offset'):
#     #         self.conv_offset.weight.data.zero_()
#     #         self.conv_offset.bias.data.zero_()


#     def forward(self, x1, x2=None):
#         # self.conv1.weight.data = self.conv1.weight * self.mask
#         # offset_range = x1.size(-1) // 4

#         if x2 == None:
#             x2 = x1

#         t = self.conv_offset(x1)
#         # t = diff_conv(t, x2, self.conv_offset)
        
#         def traj_encode(x, t, f):
#             x = f['conv1'](x, t)
#             x = diff_conv(x, x2, f['conv1'])
#             x = f['conv2'](x)
#             return x

#         m = traj_encode(x1, t, self.conv_m)
#         return m

class Decouple(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):

        super(Decouple, self).__init__()
        assert kernel_size > 1, "Not support 1 * 1 Conv"

        # self.conv1 = nn.Conv2d(in_channels, in_channels * D, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=in_channels, bias=False)
        self.kernel_size = kernel_size
        # self.conv_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,  kernel_size=3, padding=1, stride=stride, dilation=dilation, bias=False)
        # self.conv_offset = nn.Conv2d(in_channels, 4 * 2 * kernel_size * kernel_size,  kernel_size=3, padding=1, stride=stride, dilation=dilation, bias=True)
        
        # self.conv_o = InvBottle(in_channels, 4 * 2 * kernel_size * kernel_size,  kernel_size=3, padding=1, stride=stride, dilation=dilation, bias=True)
        self.conv_o = nn.Conv2d(in_channels, 4 * 2 * kernel_size * kernel_size,  kernel_size=3, padding=1, stride=stride, dilation=dilation, bias=True)
        self.conv_m = MOP.DeformConv2d(in_channels, in_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, groups=in_channels, deform_groups=4)
        # self.conv_m = MOP.DeformConv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, deform_groups=4)
        self.conv_r = InvBottle(in_channels + 4 * 2 * kernel_size * kernel_size, out_channels, kernel_size=1, padding=0, stride=stride, dilation=dilation, activation='tanh', bias=False)
        
    def forward(self, x, x2=None):
        # self.conv1.weight.data = self.conv1.weight * self.mask
        # offset_range = x1.size(-1) // 4

        if x2 == None:
            x2 = x.clone()

        # t = self.conv_o(x)
        # x = self.conv_m(x, t)
    
        t = self.conv_o(x) - F.conv2d(input=x2, weight=self.conv_o.weight.sum((2, 3), keepdim=True), groups=self.conv_o.groups)
        x = self.conv_m(x, t) - F.conv2d(input=x2, weight=self.conv_m.weight.sum((2, 3), keepdim=True), groups=self.conv_m.groups)

        # w = torch.ones_like(self.conv_o.weight[..., 0:1, 0:1])
        # t = self.conv_o(x) - self.conv_o((x > 0).float()) * F.conv2d(input=x2, weight=w,  groups=self.conv_o.groups)

        # w = torch.ones_like(self.conv_m.weight[..., 0:1, 0:1])
        # x = self.conv_m(x, t) - self.conv_m((x > 0).float(), t) * F.conv2d(input=x2, weight=w, groups=self.conv_m.groups)
        t = t * (x2.sum(dim=1, keepdim=True) > 0)
        x = torch.cat([x, t], dim=1)
        x = self.conv_r(x)

        return x

class RecurrentDecoupleLayer(nn.Module):
    """
    Layer comprised of a convolution followed by a recurrent convolutional block.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(self, in_channels, out_channels,
                kernel_size=3, stride=1, activation_ff="relu", activation_rec=None, norm=None, **kwargs):
        super(RecurrentDecoupleLayer, self).__init__()

        self.down = DownSample(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        
        self.dec = Decouple( out_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1)
        # self.dec = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2,)

        self.rec = ConvGRU(out_channels, out_channels, kernel_size=3, activation=activation_rec, conv_func=InvBottle)
        self.res = ResidualBlock(out_channels, out_channels, 1, activation=activation_ff, norm=norm)

    def forward(self, x, state):
        x = self.down(x)
        t = self.dec(x)
        t, state = self.rec(t, state)
        t = self.res(t)
        return x, t, state

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

class MultiResUNetRecurrent(nn.Module):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """
    def __init__(self, 
                base_num_channels,
                num_encoders,
                num_residual_blocks,
                num_output_channels,
                norm,
                num_bins,
                kernel_size=5,
                channel_multiplier=2,
                activations_ff="relu",
                activations_rec=None,
                activation_out="tanh"):

        super(MultiResUNetRecurrent, self).__init__()
        self.__dict__.update({
            'num_encoders': num_encoders,
            'num_residual_blocks': num_residual_blocks,
            'num_output_channels': num_output_channels,
            'norm': norm,
            'num_bins': num_bins,
            'kernel_size': kernel_size,
            'ff_act': activations_ff,
            'rec_act': activations_rec,
            'final_activation': activation_out
        })
        assert num_output_channels > 0

        self.encoder_input_sizes = [
            int(base_num_channels * pow(channel_multiplier, i)) for i in range(num_encoders)
        ]
        self.encoder_output_sizes = [
            int(base_num_channels * pow(channel_multiplier, i + 1)) for i in range(num_encoders)
        ]

        self.max_num_channels = self.encoder_output_sizes[-1]

        self.num_states = num_encoders
        self.states = [None] * self.num_states
        
        self.encoders = self.build_encoders()
        self.decoders = self.build_decoders()
        self.preds = self.build_multires_prediction()

    def build_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            stride = 2
            encoders.append(
                RecurrentDecoupleLayer(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    activation_ff=self.ff_act,
                    activation_rec=self.rec_act,
                    norm=self.norm,
                )
            )
        return encoders

    def build_multires_prediction(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for i, input_size in enumerate(decoder_output_sizes):
            preds.append(
                nn.Sequential(
                    InvBottle(
                        input_size,
                        self.num_output_channels,
                        1,
                        bias=False,
                        activation=self.final_activation
                    ),
                    # MedianPool2d(
                    #     kernel_size=3,
                    #     stride=1,
                    #     padding=1, same=False
                    # ),
                )
                
            )
        return preds

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            input_size = input_size if i == 0 else input_size * 2 + 2
            decoders.append(
                nn.Sequential(
                    InvBottle(
                        input_size,
                        output_size,
                        self.kernel_size,
                        padding=self.kernel_size//2,
                        bias=False,
                        activation=self.ff_act,
                        norm=self.norm),
                    MedianPool2d(
                        kernel_size=3,
                        stride=1,
                        padding=1, same=False
                    ),
                )
            )
        return decoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        traj = []
        for i, encoder in enumerate(self.encoders):
            x, t, state = encoder(x, self.states[i])
            traj.append(t)
            self.states[i] = state

        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            if i > 0:
                t = B.skip_concat(traj[-i - 1], t)
                t = B.skip_concat(p, t)
            t = F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
            t = decoder(t)
            p = pred(t)
            predictions.append(p)

        return predictions

class Model(MultiResUNetRecurrent):
    """
    Recurrent version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    """
    def __init__(self,
                norm=None,
                base_num_channels=32, 
                num_encoders=4,
                num_residual_blocks=2,
                num_output_channels=2,
                num_bins=2,
                norm_input=True,
                kernel_size=3,
                channel_multiplier=2,
                activations_ff="relu",
                activations_rec=None,
                activation_out="tanh",
                mask_output=True,
                **kwargs
                ):
        super().__init__(base_num_channels, num_encoders, num_residual_blocks, num_output_channels,
            norm, num_bins, kernel_size, channel_multiplier, activations_ff, activations_rec, activation_out)

        self.crop = None
        self.mask = mask_output
        self.norm_input = norm_input
        self.num_bins = num_bins
        self.image_padder = ImagePadder(min_size=16)

    def detach_states(self):        
        def detach(state):
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    hidden = detach(hidden)
                    tmp.append(hidden)
                return tuple(tmp)
            else:
                return state.detach()
        
        self.states = [detach(state) for state in self.states]

    def reset_states(self):
        self.states = [None] * self.num_states

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = B.CropParameters(width, height, self.num_encoders, safety_margin)

    def flow_resize(self, multires_flow, size=None):
         # upsample flow estimates to the original input resolution
        flow_list = []
        for i, flow in enumerate(multires_flow):
            scaling_h = size[0] / flow.size(-2)
            scaling_w = size[1] / flow.size(-1)
            # scaling_flow = 2 ** (self.num_encoders - i - 1)
            scaling_flow = 1
            upflow = scaling_flow * F.interpolate(
                flow, scale_factor=(scaling_h, scaling_w), mode="bilinear", align_corners=False
            )
            # upflow = self.image_padder.unpad(upflow)
            flow_list.append(upflow)
        
        return flow_list

    def forward(self, x, size):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """
        
        # image padding
        x = self.image_padder.pad(x).contiguous()
        x = gaussian_blur(x, kernel_size=3)

        # normalize input
        if self.norm_input:
           x = B.nonzero_normalize(x)
        
        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass
        multires_flow = super().forward(x)
        multires_flow = self.flow_resize(multires_flow, size)
        return {"flow": multires_flow}