# STDC Format:

# Json file:
# { "<videoid/frameid1>": { 'height,width': ,
#                           'annotations': [{"bounding box" : ,
#                                               'object_caption': [list of strings] ,
#                                               "vocab_index": [list of list of int data type], 
#                                           }
#                                           ]
#                          }
# 
# }
# 
# UITG.mp4/000101.jpg -> 000099, 97, 95, 93, 91...85, 103, 105, 107, 

# Each datapoint 
# 1. Keyframe and Context frames (nearby, sparse sampling)
# 2. Bounding boxes # (B,4) torch.floatTensor
# 3. Caption Indices # (B,10) torch.LongTensor

