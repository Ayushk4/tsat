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

#----------------------------------------
#--------- Common imports ---------------
#----------------------------------------
from common.utils.masks import *


