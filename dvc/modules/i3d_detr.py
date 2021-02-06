#----------------------------------------
#--------- I3D and DETR ----------------
#----------------------------------------

"""
Inputs:
    - A set of frames (fixed number of frames) and keyframe index

Outputs:
    - Regions and captions (a single caption for each frame for the time being)
    
Model:
    - Backbone: I3D (temporally pooled)
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

from <> import I3D
from <> import DETR
# import head

class I3D_DETR:

    def __init__(self, config):

        self.backbone = I3D(config)

        self.DETR = DETR(config)

        self.head = 

    def forward(self, frames, keyframe_idx):
        pass
