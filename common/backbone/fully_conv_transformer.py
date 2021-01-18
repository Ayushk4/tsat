#------------------------------------------------------------
#--------- Full Convolutional Temporal Transformers ---------
#--------- Written by Nikhil Shah and Ayush Kaushal----------
#------------------------------------------------------------

"""
Fully convolutional temporal transformers explained:-

Inputs:   Multiple frames from a single video, the central
          frame being the keyframe

Outputs:  The temporally contextualized features of the 
          keyframe

Method:   Each transformer block consists of two stages
          1. Convolutions
          2. Temporal Self-Attention

          - The parameters are shared for each frame.
          - The goal is to calculate deep features of each image
            and contextualize them temporally
          - The overall architecture of convolutions can resemble to
            that of any generic architecture, ResNet for example
          - For the temporal self-attention we'll convert feature maps into
            queries using 1x1 convolutions and then take matmul of query and key
            to find the attention maps.
          - If the feature maps are of size: WxHxC, then the attention maps will be
            of the same size.
          - Let's say there are n feature maps: f1, f2, ..., fn. For calculating the
            attended feature maps for let's say f2, we will generate n attention maps
            (analogous to n attention scores in case of words). Let these attention maps
            be a1, a2, ..., an. The resultant feature maps will be element wise multiplications
            of feature maps and attention maps. i.e.
            f2' = a1*f1 + a2*f2 + ... + an*fn
          - Once we obtain the attended features, the resultant features are calculated as
            f2'' = γ*f2' + f2, where γ is a learnable scalar
"""

#----------------------------------------
#--------- Library imports --------------
#----------------------------------------

import torch
import torch.nn as nn
from opt_einsum import contract

#----------------------------------------
#--------- Common imports ---------------
#----------------------------------------
import math
from typing import Type, Any, Callable, Union, List, Optional

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__( self, inplanes: int, planes: int,
        stride: int = 1, downsample: Optional[nn.Module] = None,
        groups: int = 1, base_width: int = 64, dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, *args, **kwargs):

        """
        For videos, x won't be a batch of single images.
        Rather, it will be a batch with each example containing
        multiple frames

        Inputs:

            1. x:
                - Shape: N * F * C * W * H

        We'll unpack x into the shape:
                        (N * F) * C * W * H
        """

        # unpack x into images
        x_size = x.size()
        x = x.view(-1, *x_size[-3:])
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        # pack again to frames
        out_size = out.size()
        out = out.view(*x_size[:2], *out_size[-3:])

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__( self, inplanes: int, planes: int,
        stride: int = 1, downsample: Optional[nn.Module] = None,
        groups: int = 1, base_width: int = 64, dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:

        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, *args, **kwargs):
        # unpack x into images
        x_size = x.size()
        x = x.view(-1, *x_size[-3:])

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out_size = out.size()
        out = out.view(*x_size[:2], *out_size[-3:])

        return out

class TemporalSelfAttention(nn.Module):

    def __init__(self, config, num_channels):
        super(TemporalSelfAttention, self).__init__()

        # construct keys, queries and values
        self.key_layer = conv1x1(num_channels, num_channels)
        self.query_layer = conv1x1(num_channels, num_channels)
        self.value_layer = conv1x1(num_channels, num_channels)

        # dropout
        self.dropout = nn.Dropout(config.ATTENTION_DROPOUT)

        # we have to construct a learnable variable
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, context_frames, attention_mask, output_attention_probs=False):

        """
        Inputs:
            1. context_frames
                - Shape: N * F * C * H * W

        We have to calculate the temporal self attention

        Steps:
        1. Convert into keys, queries and values
        2. Find the attention maps by taking the interaction of keys and queries using matmul
        3. Take hadamard product of attention maps with values to obtain attended features
        """

        # convert into keys, queries and values
        size = context_frames.size()

        # find out the width and height
        channels, height, width = size[-3:]

        # unpack context frames
        context_frames_unpacked = context_frames.view(-1, channels, height, width)

        # we can directly view in the same size because 1x1 conv does not alter shape
        key_frames = self.key_layer(context_frames_unpacked).view(*size)
        query_frames = self.query_layer(context_frames_unpacked).view(*size)
        value_frames = self.value_layer(context_frames_unpacked).view(*size)

        # calculate the attention maps
        # and divide by the total number of summations
        attention_maps = contract('tfcxy, tgcyz -> tfg', query_frames, key_frames.transpose(-1,-2))
        attention_maps = attention_maps / math.sqrt(channels*height*width*width) 

        # apply the attention mask, calculated in the model file
        attention_maps = attention_maps + attention_mask

        # next take softmax over frames
        attention_probs = nn.Softmax(dim=-1)(attention_maps)

        # attention dropout as used in the original bert model
        attention_probs = self.dropout(attention_probs)

        # calculate the product of attention_probs and value_frames
        # Shape of attention_probs: N * F * F
        # Shape of value_frames: N * F * C * H * W
        contextualized_frames = contract('tfg,tgchw->tfchw', attention_probs, value_frames)

        # add the skip connection
        contextualized_frames = self.gamma * contextualized_frames + context_frames

        if output_attention_probs:
            return contextualized_frames, attention_probs
        else:
            return contextualized_frames


class FullyConvTransformer(nn.Module):

    def __init__(self, config, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(FullyConvTransformer, self).__init__()

        # Config
        self.config = config

        # Norm layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Set the params to keep track of
        self.inplanes = 64
        self.dilation = 1

        # Each elem in `replace_stride_with_dilation`
        # representes whether to replace the 2x2 stride
        # with dilated convolutions instead
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Initial conv layer to process the feature maps
        self.initial_conv_layer = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        # Next, we create 4 resnet blocks with temporal self attention
        self.tsa_resblock_1 = self._make_tsa_resblock(block, 64, layers[0])
        self.tsa_resblock_2 = self._make_tsa_resblock(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.tsa_resblock_3 = self._make_tsa_resblock(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.tsa_resblock_4 = self._make_tsa_resblock(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # Create a ModuleList of the ResBlocks
        self.all_layers = nn.ModuleList([self.tsa_resblock_1, self.tsa_resblock_2, self.tsa_resblock_3, self.tsa_resblock_4])

        # init Kaiming Normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_tsa_resblock(self, block, planes, blocks, stride=1, dilate=False):

        # set the norm_layer to a class specified in init
        # Note that this is not an object but the class name only
        norm_layer = self._norm_layer

        # Create a None object for downsample layer
        downsample = None

        # Just in case we use dilation
        previous_dilation = self.dilation

        # If using dilation multiply the dilation by stride
        # and set stride to 1
        if dilate:
            self.dilation *= stride
            stride = 1

        # For the first block in a ResBlock, 
        # we need to create the downsample object
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        # Now we create the components in a tsa_resblock
        layers = []

        # First up is the temporal self attention block
        # The gamma multiplication is in the TSA itself
        layers.append(TemporalSelfAttention(self.config.NETWORK.TSA, 
                                            num_channels=self.inplanes))

        # Next, is the general ResNet block
        # The first block which has stride 2 and downsample layer
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        # The first block expands the feature maps in num of channels
        # and shrink the same in width and height
        self.inplanes = planes * block.expansion

        # Some more resnet blocks
        for _ in range(1, blocks):
            # Check if we have to append TSA layer before each block
            if self.config.NETWORK.TSA_EVERY_BLOCK:
                layers.append(TemporalSelfAttention(self.config.NETWORK.TSA, 
                                                num_channels=self.inplanes))

            # Append the other resnet blocks
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.ModuleList(layers)

    def forward(self, context_frames, keyframe_index, attention_mask):

        """
        Inputs:
            1. context_frames:
                - Shape: N * F * C * H * W
                - Images for all the sampled timestamps

            2. keyframe_index:
                - Shape: N * 1
                - Index of the keyframe whose features we have to return

            Where,
            N = batch_size
            F = total number of frames
            C = channels
            H = height
            W = width

        Flow for a single `tsa_resblock`:
            Assuming input shape: N * F * C * H * W

            Module      |       Output Shape
            --------------------------------
            1. TSA      |       N * F * C * H * W
            2. Conv     |       N * F * C' * H/2 * W/2
            3. TSA (Op.)|       N * F * C' * H/2 * W/2
            4. Conv     |       N * F * C' * H/2 * W/2
            5. TSA (Op.)|       N * F * C' * H/2 * W/2
            6. Conv     |       N * F * C' * H/2 * W/2
            7. TSA (Op.)|       N * F * C' * H/2 * W/2
            8. Conv     |       N * F * C' * H/2 * W/2

            (Op.) represents optional
            Assuming the stride=2 for first block and 1 for others

        Finally, we need to return either the features for all frames
        or for the keyframe only depending on the config flag
        """

        # Apply the initial conv layers
        N, F, C, H, W = context_frames.size()

        context_frames = context_frames.view(-1, C, H, W)
        context_frames = self.initial_conv_layer(context_frames)
        _, C_, H_, W_ = context_frames.size()
        context_frames = context_frames.view(N, F, C_, H_, W_)

        for tsa_resblock in self.all_layers:
            for module in tsa_resblock:
                context_frames = module(context_frames, attention_mask)

        if self.config.NETWORK.RETURN_ALL_FEATURES:
            return context_frames
        else:
            return context_frames[:,keyframe_index,:,:,:]



