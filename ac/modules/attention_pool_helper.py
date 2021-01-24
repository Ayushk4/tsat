import torch
import torchvision 
from torch import nn

class AttentionPool(nn.Module):

    def __init__(self, input_vec_dimensions=1024):
        super(AttentionPool, self).__init__()

        # Transformation layer
        self.transformation = nn.Linear(input_vec_dimensions, 1)
        self.softmax = nn.Softmax(-2)

    def forward(self, temporal_feats, attention_mask):

        # Shape: BS x F x input_vec_dimensions
        attention_scores = self.transformation(temporal_feats)

        # mask according to the `attention_mask`
        attention_scores -= 10000 * (attention_mask.unsqueeze(-1).float())

        # take the softmax: B x F
        attention_scores = self.softmax(attention_scores)

        # matmul
        # B x F x 1, B x F x D --> B x D
        pooled_temporal_feats = torch.matmul(attention_scores.transpose(-1,-2), temporal_feats).squeeze(-2)

        return pooled_temporal_feats


