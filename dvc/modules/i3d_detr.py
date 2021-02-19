#----------------------------------------
#---- Backbone, DETR and Captioning------
#----------------------------------------

"""
Inputs:
    - A set of frames (fixed number of frames) and keyframe index

Outputs:
    - Regions and captions (a single caption for each frame for the time being)
    
Model:
    - Backbone: I3D (temporally pooled) or any other backbone
        - Outputs: Features (like the inputs of DETR: B x C x H x W)

    - DETR
        - Features of regions of interest (B x N x D)

    - Head
        - Bounding box coordinates prediction
        - Useful ROI or not
        - Captioning head (Only input those rois with IOU > certain threshold)

Loss:
    - DETR object detection loss (bipartite matching loss)
    - ....
    - Captioning loss (NLL)
"""
import torch
import torch.nn as nn

from common.backbone.i3d import I3ResNet
from .detr_pos_enc_helper import *
from .transformer_helper import Transformer

class DetrCaptioning(nn.Module):

    def __init__(self, config):
        super(DetrCaptioning, self).__init__()

        # Unpack some items fromm config
        hidden_dim = config.NETWORK.TRANSFORMER.HIDDEN_DIM
        num_classes = config.NETWORK.NUM_CLASSES
        self.aux_loss = config.NETWORK.USE_AUX_LOSS

        # First comes the backbone
        self.backbone = eval(config.NETWORK.BACKBONE_MODULE)(config.NETWORK.BACKBONE)

        # Next is the positional encoding
        self.position_encoding = eval(config.NETWORK.POSITIONAL_ENCODING_TYPE)(config.NETWORK.POSITIONAL_ENCODING)

        # Next, the transformer
        # The projection layer
        self.input_proj = nn.Conv2d(config.NETWORK.BACKBONE.NUM_CHANNELS, hidden_dim, kernel_size=1)
        # To input into the decoder of the transformer, the learnable object queries
        self.query_embed = nn.Embedding(config.NETWORK.TRANSFORMER.NUM_QUERIES, hidden_dim)
        self.transformer = Transformer(config.NETWORK.TRANSFORMER)

        # Head things
        # BBOX coordinates prediction
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # object class prediction
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        # captioning head
        # TODO

    def forward(self, frames, keyframe_idx, **kwargs):

        backbone_features = self.backbone(frames, keyframe_idx)
        position_encodings = self.position_encoding(backbone_features)

        #TODO: Create a mask (Shape: N x H x W)
        hidden_states = self.transformer(self.input_proj(backbone_features), mask, self.query_embed.weight, position_encoding)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        """
        #Ommiting the aux_loss for now
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        """
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


