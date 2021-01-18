#----------------------------------------
#--------- Library Imports --------------
#----------------------------------------
import torch
import torchvision 
from torch import nn

#----------------------------------------
#--------- FC Transformer Backbone ------
#----------------------------------------
from common.backbone.fully_conv_transformer import FullyConvTransformer, BasicBlock, Bottleneck

class FCTransformerForAC(nn.Module):

    def __init__(self, config, block, layers, **kwargs):

        super(FCTransformerForAC, self).__init__()

        """
        Here, we'll create the learnable positional embeddings, 
        classification token, attention masks (during forward)
        and classifier mlp and pooling layer
        """


        # First create the learnable [CLS] token
        self.cls_encoding = nn.Parameter(torch.rand((1,1, *config.NETWORK.IMAGE_SPAT_DIMENSION)))

        # Create the adaptive pool layer and final_mlp
        self.backbone_output_dimension = 512 * block.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_mlp = nn.Sequential([
            nn.Linear(self.backbone_output_dimension, config.NETWORK.FINAL_MLP_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(config.NETWORK.FINAL_MLP_DROPOUT),
            nn.Linear(config.NETWORK.FINAL_MLP_HIDDEN, config.NETWORK.NUM_CLASSES)
        ])

        # Create Positional Encodings
        # TODO: add positional encodings either here or to TSA
        # Latter is preferred

        # Now the backbone
        self.backbone = FullyConvTransformer(config, block, layers, **kwargs)

    def forward(self, context_frames, keyframe_index, frames_pad_mask):

        """
        Inputs:
            context_frames:
                - Shape: B * F * C * H * W
            keyframe_index:
                - Shape: B * 1
                    - For classification we keep it zero
            frames_pad_mask:
                - Shape: B * F

        We need to append the classification token and also update the
        attention mask accordingly
        """
        B, F, C, H, W = context_frames.size()

        # create the cls token
        assert self.cls_encoding.size() == (1,1,C,H,W)
        cls_token = self.cls_encoding.expand(B, 1, C, H, W)

        # concatenate cls token with the frames
        context_frames = torch.cat((cls_token, context_frames), dim=1)

        # create the attention mask
        # First, add a False for the CLS token
        device = frames_pad_mask.device()
        frames_pad_mask = torch.cat((torch.zeros((B,1),dtype=torch.bool).to(device),
                                    frames_pad_mask), 1)

        # Now create the additive attention mask
        # Current shape: N * (F+1)
        # Final shape we need: N * (F+1) * (F+1) (repeated F times)
        # -10000 where there is True, else 0
        attention_mask = frames_pad_mask.unsqueeze(1)
        attention_mask = attention_mask.expand(B,(F+1),(F+1))
        attention_mask = attention_mask.float() * -10000

        # pass through the transformer model
        classification_spatial_features = self.backbone(context_frames, 0, attention_mask)

        # apply the classification layers
        features = self.avgpool(classification_spatial_features)
        logits = self.final_mlp(features)

        return logits


def _resnet(config, arch, block, layers, **kwargs):

    model = FCTransformerForAC(config, block, layers, **kwargs)
    return model


def resnet18_transformer(config, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(config, 'resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_transformer(config, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(config, 'resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_transformer(config, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(config, 'resnet50', Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_transformer(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(config, 'resnet101', Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_transformer(config, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(config, 'resnet152', Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d_transformer(config, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(config, 'resnext50_32x4d', Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d_transformer(config, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(config, 'resnext101_32x8d', Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2_transformer(config, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(config, 'wide_resnet50_2', Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2_transformer(**kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(config, 'wide_resnet101_2', Bottleneck, [3, 4, 23, 3], **kwargs)
