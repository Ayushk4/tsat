import math

import torch
from torch.nn import ReplicationPad3d
import torchvision 

from common.i3d_utils import inflator as inflate

bb_to_tv_function = {"res18": torchvision.models.resnet18,
                    "res34": torchvision.models.resnet34,
                    "res50": torchvision.models.resnet50,
                    "res101": torchvision.models.resnet101
                }
bb_out_dims = {"res18": 512, # Backbone_out_dims
                "res34": 512,
                "res50": 2048,
                "res101": 2048
            }

class I3ResNet(torch.nn.Module):
    def __init__(self, config):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet, self).__init__()
        frame_nb=16
        conv_class = config.NETWORK.CONV_CLASS

        resnet2d = bb_to_tv_function[config.NETWORK.BACKBONE](pretrained=config.NETWORK.BACKBONE_LOAD_PRETRAINED)
        self.conv_class = conv_class

        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=3, time_padding=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2)

        self.layer1 = inflate_reslayer(resnet2d.layer1, config.NETWORK.BACKBONE)
        self.layer2 = inflate_reslayer(resnet2d.layer2, config.NETWORK.BACKBONE)
        self.layer3 = inflate_reslayer(resnet2d.layer3, config.NETWORK.BACKBONE)
        self.layer4 = inflate_reslayer(resnet2d.layer4, config.NETWORK.BACKBONE)

        if conv_class:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            self.classifier = torch.nn.Conv3d(
                in_channels=bb_out_dims[config.NETWORK.BACKBONE],
                out_channels=config.NETWORK.NUM_CLASSES,
                kernel_size=(1, 1, 1),
                bias=True)
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            self.avgpool = inflate.inflate_pool(
                resnet2d.avgpool, time_dim=final_time_dim)
            self.fc = inflate.inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.conv_class:
            x = self.avgpool(x)
            x = self.classifier(x)
            x = x.squeeze(3)
            x = x.squeeze(3)
            x = x.mean(2)
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x


def inflate_reslayer(reslayer2d, backbone):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d, backbone)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d, backbone):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate.inflate_conv(
            bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        if backbone not in ["res18", "res34"]:
            self.conv3 = inflate.inflate_conv(
                bottleneck2d.conv3, time_dim=1, center=True)
        else:
            self.conv3 = None

        if backbone not in ["res18", "res34"]:
            self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        else:
            self.bn3 = None

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride)
        else:
            self.downsample = None

        assert (self.conv3 != None and self.bn3 != None) or (self.conv3 == None and self.bn3 == None)
        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.conv3 != None:
            out = self.conv3(out)
        if self.bn3 != None:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True),
        inflate.inflate_batch_norm(downsample2d[1]))
    return downsample3d
