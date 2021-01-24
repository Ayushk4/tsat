import torch
import torchvision 
from torch import nn
from common.utils.positional_enc import PositionalEncoding

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

class VideoTransformer(nn.Module):

    def __init__(self, config):
        super(VideoTransformer, self).__init__()
        
        # Store necessary data from config
        self.backbone_name = config.NETWORK.BACKBONE
        self.backbone_load_pretrained = config.NETWORK.BACKBONE_LOAD_PRETRAINED
        assert type(self.backbone_load_pretrained) == bool
        assert self.backbone_name in bb_to_tv_function.keys()

        self.bb_out_dims = bb_out_dims[self.backbone_name]
        self.temporal_mlp_hidden = config.NETWORK.TEMPORAL_MLP_DIMS
        self.temporal_mlp_activation = config.NETWORK.TEMPORAL_MLP_ACTIVATION
        
        self.transformer_feed_spatial = config.NETWORK.PASS_SPATIAL_TO_TRANSFORMER

        self.transformer_dims = config.NETWORK.TRANSFORMER_DIMS
        self.transformer_heads = config.NETWORK.TRANSFORMER_HEADS
        self.transformer_encoder_cnt = config.NETWORK.TRANSFORMER_ENCODER_CNT
        self.transformer_dropout = config.NETWORK.TRANSFORMER_DROPOUT
        self.transformer_feedforward_dims = config.NETWORK.TRANSFORMER_FEEDFORWARD_DIMS

        self.positional_dropout = config.NETWORK.POSITIONAL_DROPOUT

        self.num_classes = config.NETWORK.NUM_CLASSES

        # Video Backbone
        # TODO: Crosscheck whether backbone is being loaded from pretrained
        backbone_full = bb_to_tv_function[self.backbone_name](pretrained=self.backbone_load_pretrained)
        self.backbone = nn.Sequential(*(list(backbone_full.children())[:-2]))

        # Backbone to transformer
        # TODO: Add 1D Conv before Temp pool 
        self.bb_to_temp_pool = nn.Sequential(list(backbone_full.children())[-2],)
        self.bb_to_temporal = nn.Sequential(nn.Linear(self.bb_out_dims, self.temporal_mlp_hidden),
                                            eval("nn."+self.temporal_mlp_activation)(),
                                            nn.Linear(self.temporal_mlp_hidden, self.transformer_dims)
                                        )
        if self.transformer_feed_spatial:
            self.bb_to_spatial = nn.Linear(self.bb_out_dims, self.transformer_dims)

        # Positional and Segment Encoding
        # TODO: shift to learnable 1D positional encoding and have different instantiations
        #       for temporal and spatial sequence.
        # TODO: DETR adds positional encoding at every timestep (should we do the same)
        self.positional_encoding = PositionalEncoding(self.transformer_dims,
                                                    self.positional_dropout,
                                                )

        # Transformer
        encoder_layer_single = nn.TransformerEncoderLayer(self.transformer_dims, self.transformer_heads,
                                                    dim_feedforward=self.transformer_feedforward_dims,
                                                    dropout=self.transformer_dropout
                                            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer_single,
                                                        self.transformer_encoder_cnt
                                            )

        # [CLS] and [SEP] learnable vectors
        # TODO: Better Initialization
        self.cls = nn.Parameter(torch.rand(1, 1, self.transformer_dims))
        self.end = nn.Parameter(torch.rand(1, 1, self.transformer_dims))
        if self.transformer_feed_spatial:
            self.sep = nn.Parameter(torch.rand(1, 1, self.transformer_dims))
        
        # Segment Embeddings: [SPAT] and [TEMP]
        if self.transformer_feed_spatial:
           self.spat_seg_emb = nn.Parameter(torch.rand(1, 1, self.transformer_dims))
        self.temp_seg_emb = nn.Parameter(torch.rand(1, 1, self.transformer_dims))

        # Tagging layer
        self.output_tagger = nn.Linear(self.transformer_dims, self.num_classes)
        #self.output_tagger = nn.Sequential(
        #        nn.Linear(self.transformer_dims, config.NETWORK.FINAL_MLP_HIDDEN),
        #        nn.ReLU(inplace=True),
        #        nn.Dropout(config.NETWORK.FINAL_MLP_DROPOUT),
        #        nn.Linear(config.NETWORK.FINAL_MLP_HIDDEN, self.num_classes)
        #    )

        # TODO: Set Default Initiliazer.

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
            
            ********************
            ******* NOTE *******
            ********************
            The pad mask should also take into consideration the spatial sequence (mostly 49 length)
            before the temporal and the special [CLS] and [SEP] tokens.
            This is taken into consideration inside this function.

        """
        # TODO:
        # Make sure the torchvision.models preprocessing requirements are correctly met.
        B, T, C, H, W = frames.shape
        assert frames_pad_mask.shape[1] == frames.shape[1]

        # Pass frames through backbone
        # TODO: Introduce padding only in the transformer, flatten out the portion
        #       in backbone into 4D input for reduced computation
        # before passing into the backbone, unpad the frames
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

        temp_feats = self.bb_to_temp_pool(bb_feats)
        temp_feats = self.bb_to_temporal(temp_feats.view(B, T, -1))

        # pad them again

        # Spatial feats
        if self.transformer_feed_spatial:
            _, T, C_, H_, W_ = bb_feats.shape
            spat_feat_grid = bb_feats[torch.arange(B), key_frame_idx, :, :, :] # Shape: [B, C_, H_, W_]
            spat_feat_sequence = spat_feat_grid.view(B, C_, H_ * W_).permute(0,2,1)
            spat_feat_sequence = self.bb_to_spatial(spat_feat_sequence)

        # Calculate the Positional Encodings
        # Actually, the positional encoding is already added in the following function call
        temp = self.positional_encoding(temp_feats)
        if self.transformer_feed_spatial:
            spat = self.positional_encoding(spat_feat_sequence)
        
        # Expand the segment embeddings
        temp_seg_emb = self.temp_seg_emb.expand(B, temp.size(1), self.transformer_dims)
        if self.transformer_feed_spatial:
            spat_seg_emb = self.spat_seg_emb.expand(B, spat.size(1), self.transformer_dims)
        
        # Add the segment embeddings to temporal and spatial sequence
        temp += temp_seg_emb
        if self.transformer_feed_spatial:
           spat += spat_seg_emb
        
        # Expand the token dimensions
        cls_rep = self.cls.expand(B, 1, self.transformer_dims)
        end_rep = self.end.expand(B, 1, self.transformer_dims)
        if self.transformer_feed_spatial:
            sep_rep = self.sep.expand(B, 1, self.transformer_dims)

        if self.transformer_feed_spatial:
            input_vec = torch.cat([cls_rep, spat, sep_rep,
                                temp, end_rep], 1).permute(1,0,2) # Shape = [<spat_seq>+<temp_seq>+3, B, feat_size]

            device = input_vec.device
            tf_pad_mask = torch.cat([
                            torch.zeros((B, input_vec.shape[0] - frames_pad_mask.shape[1] - 1),dtype=torch.bool).to(device),
                            frames_pad_mask,
                            torch.zeros((B, 1), dtype=torch.bool).to(device)
                        ], 1)
            assert tf_pad_mask.shape == (input_vec.shape[1], input_vec.shape[0])
        else:
            input_vec = torch.cat([cls_rep, temp, end_rep], 1).permute(1,0,2) # Shape = [<temp_seq>+3, B, feat_size]

            device = input_vec.device
            tf_pad_mask = torch.cat([
                            torch.zeros((B, 1), dtype=torch.bool).to(device),
                            frames_pad_mask,
                            torch.zeros((B, 1), dtype=torch.bool).to(device)
                        ], 1)
            assert tf_pad_mask.shape == (input_vec.shape[1], input_vec.shape[0])

        # TODO: Double check the pad mask (if this is wrong the model won't work)
        output_vectors = self.transformer_encoder(src=input_vec, src_key_padding_mask=tf_pad_mask) # Shape: [<spat_seq>+<temp_seq>+3, B, feat_size]

        # Tagger layer 
        return self.output_tagger(output_vectors[0, :, :])


