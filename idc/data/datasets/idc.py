"""
STDC Format

{"50N4E.mp4/000680.png":
            {"split": "train",
            "max_frames": 200,
            "video": "50N4E.mp4",
            "frame": "000680.png",
            "captions": [
                {"bbox": [63.4903678894043,
                    8.207611083984375,
                    192.93972778320312,
                    264.02484130859375],
                "caps": [
                    [12, 46, 71, 14, 81, 81, 81, 81, 81, 81, 81, 81]
                    ],
                "type": "person"
                }
                {"bbox": [68.13486896878712,
                    124.97725912325058,
                    90.49948044399594,
                    144.51899461451148],
                "caps": [
                    [14, 29, 57, 36, 66, 46, 81, 81, 81, 81, 81, 81],
                    [14, 68, 12, 46, 24, 71, 81, 81, 81, 81, 81, 81]
                    ],
                "type": "dish"}
            ]
            },
            ...

            ...

            10 * 4
            
}

"""

# Each datapoint 
# 1. Keyframe and Context frames (nearby, sparse sampling)
# 2. Bounding boxes # (B,4) torch.floatTensor
# 3. Caption Indices # (B,10) torch.LongTensor

from PIL import Image

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torchvision

class IDC(Dataset):

    def __init__(self, config=None, split='train'):

        super(IDC, self).__init__()

        # Load from the cache if cache is available

        # unpack the config
        # Paths and Splits
        self.root_path = config.DATASET.ROOT_PATH
        self.data_path = config.DATASET.DATA_PATH
        self.dataset_name = config.DATASET.DATASET_NAME
        self.use_toy_version = config.DATASET.TOY
        self.annotations_path = os.path.join(config.DATASET.ANNOTATIONS_PATH, split + ".json")
        self.vocab_path = config.DATASET.VOCAB_PATH
        self.split = split

        # Technical details
        self.resizer_transform = lambda x: transforms.functional.resize(x, (224,224))
        self.normalizer_transform = transforms.Compose([
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])


        # find all the annotations
        self.dataset = self.load_annotations()

        if self.use_toy_version:
            self.dataset = self.dataset[:config.DATASET.TOY_SAMPLES]

    def load_annotations(self):
        annotations = json.load(open(self.annotations_path))

        database = []

        # preprocess sampling rates and other stuff
        for k,v in annotations.items():
            
            # store all the existing information
            data_point = v
            
            database.append({'path': data_point['path'], 'caps': data_point['caps']})

        return database

    def collapse_bboxes(self, bboxes, caps, cap_text):
        assert self.split != "train"


    def preprocess_frames_and_boxes(self, image, captions_with_bb):

        image = image.float()
        image = image / 255.0

        height, width = image.shape[1], image.shape[2]

        if self.split.lower() == 'train':
            # Convert boxes to tensor
            boxes = torch.stack([torch.tensor([d[2]['x'], d[2]['y'],
                                            d[2]['x'] + d[2]['width'],
                                            d[2]['y'] + d[2]['height']]
                                            )
                                for d in captions_with_bb]
                            ).float()
        else:
            mean_list = lambda caption_grp: [sum([d[2]['x'] for d in caption_grp])/len(caption_grp),
                                     sum([d[2]['y'] for d in caption_grp])/len(caption_grp),
                                     sum([d[2]['x'] + d[2]['width'] for d in caption_grp])/len(caption_grp),
                                     sum([d[2]['y'] + d[2]['height'] for d in caption_grp])/len(caption_grp)
                                ]
            boxes = torch.stack([torch.tensor(mean_list(caption_grp))
                                for caption_grp in captions_with_bb]
                            ).float()
        # The format of boxes is [x1, y1, x2, y2]. Clips bboxes to image bounds.
        boxes = torch.tensor(clip_boxes_to_image(boxes.numpy(), height, width))

        if self.split.lower() == 'train':
            caps = torch.stack([torch.LongTensor(d[1]) for d in captions_with_bb])
            cap_text = [d[3] for d in captions_with_bb]
            assert caps.shape[0] == boxes.shape[0]
        else:
            caps = [[ddd[1] for ddd in d] for d in captions_with_bb]
            cap_text = [[ddd[3] for ddd in d] for d in captions_with_bb]
            assert len(caps) == boxes.shape[0]


        # Normalize images by mean and std.
        image = self.normalizer_transform(self.resizer_transform(image))
        # Scale BBoxes to 224,224 img:
        boxes[:, 0] = boxes[:, 0] * 224 / width
        boxes[:, 1] = boxes[:, 1] * 224 / height
        boxes[:, 2] = boxes[:, 2] * 224 / width
        boxes[:, 3] = boxes[:, 3] * 224 / height
        

        # if not self._use_bgr:
        # Convert image format from BGR to RGB.
        # image = image[:, [2, 1, 0], ...]

        return image, boxes, caps, cap_text

    def __getitem__(self, idx):

        data_point = self.dataset[idx]

        # 1. load and preprocess image : done
        # 2. bounding boxes per frames : 
        # 2. captions per frame :

        # we have to pass a tensor that contains self.sampling_count * 2 + 1 number of images

        image = torchvision.io.read_image(data_point['path'])

        image_preprocessed, boxes, caps, cap_text = self.preprocess_frames_and_boxes(
                                                            image,
                                                            data_point['caps']
                                                    )

        return image_preprocessed, boxes, caps, cap_text, self.split == "train"
                



def clip_boxes_to_image(boxes, height, width):
    """
    Clip an array of boxes to an image with the given height and width.
    Args:
        boxes (ndarray): bounding boxes to perform clipping.
            Dimension is `num boxes` x 4.
        height (int): given image height.
        width (int): given image width.
    Returns:
        clipped_boxes (ndarray): the clipped boxes with dimension of
            `num boxes` x 4.
    """
    clipped_boxes = boxes.copy()
    clipped_boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    clipped_boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return clipped_boxes

