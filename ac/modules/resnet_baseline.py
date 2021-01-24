import torch
import torchvision 
from torch import nn
from common.utils.positional_enc import PositionalEncoding
from .attention_pool_helper import AttentionPool

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


class ResnetBaseline(nn.Module):

    def __init__(self, config):
        super(ResnetBaseline, self).__init__()
        
        # Store necessary data from config
        self.backbone_name = config.NETWORK.BACKBONE
        self.backbone_load_pretrained = config.NETWORK.BACKBONE_LOAD_PRETRAINED
        self.attention_pool_dim = config.NETWORK.TRANSFORMER_DIMS
        assert type(self.backbone_load_pretrained) == bool
        assert self.backbone_name in bb_to_tv_function.keys()

        self.bb_out_dims = bb_out_dims[self.backbone_name]
        self.num_classes = config.NETWORK.NUM_CLASSES

        # Video Backbone
        # TODO: Crosscheck whether backbone is being loaded from pretrained
        backbone_full = bb_to_tv_function[self.backbone_name](pretrained=self.backbone_load_pretrained)
        self.backbone = nn.Sequential(*(list(backbone_full.children())[:-1]))

        # Backbone to transformer
        self.bb_to_temporal = nn.Sequential(nn.Linear(self.bb_out_dims, self.attention_pool_dim))

        # attention pooling module
        self.attention_pool = AttentionPool(input_vec_dimensions=self.attention_pool_dim)
        
        self.output_tagger = nn.Linear(self.attention_pool_dim, self.num_classes)
        #self.output_tagger = nn.Sequential(
        #        nn.Linear(self.transformer_dims, config.NETWORK.FINAL_MLP_HIDDEN),
        #        nn.ReLU(inplace=True),
        #        nn.Dropout(config.NETWORK.FINAL_MLP_DROPOUT),
        #        nn.Linear(config.NETWORK.FINAL_MLP_HIDDEN, self.num_classes)
        #    )

    def forward(self,
                frames: torch.Tensor,
                key_frame_idx: torch.LongTensor,
                frames_pad_mask: torch.BoolTensor
            ):
        """
        Parameters
        ----------
        frames: Tensor[B, T, C, H, W]:
            video frames (preprocessed and batched)

        key_frame_idx: LongTensor[B]:
            keyframe index of each batch of frames

        frames_pad_mask: BoolTensor[B, T]:
            `frames_pad_mask` provides specified elements in the key
            to be **ignored** by the attention.
            Given a BoolTensor is provided, the positions with the value
            of True will be ignored while the position with the value of
            False will be unchanged.
        """

        B, T, C, H, W = frames.shape
        assert frames_pad_mask.shape[1] == frames.shape[1]

        """
        Method to avoid any problem with batchNorm,

        We first unpad the frames into their true size,
        then pass them through the backbone that contains the batchNorm layer,
        we then pad the obtained spatial features again with zeros
        """
        frames = frames.view(-1, C, H, W)
        frames_pad_mask_unpacked = frames_pad_mask.contiguous().view(-1)
        true_frames = frames[~frames_pad_mask_unpacked]
        true_bb_feats = self.backbone(true_frames)

        # pad the bb feats
        # create the zero_tensor
        true_bb_feats_size = true_bb_feats.size()
        bb_feats = torch.zeros(B*T, *true_bb_feats_size[1:]).to(true_bb_feats.device)
        bb_feats[~frames_pad_mask_unpacked] += true_bb_feats
        bb_feats = bb_feats.contiguous().view(B, T, *true_bb_feats_size[1:])

        temp_feats = self.bb_to_temporal(bb_feats.view(B, T, -1))

        # attention pool
        temp_feats = self.attention_pool(temp_feats, frames_pad_mask)

        return self.output_tagger(temp_feats)


