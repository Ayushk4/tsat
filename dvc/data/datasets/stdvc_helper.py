#!/usr/bin/env python3

import os
import json
from fvcore.common.file_io import PathManager
import numpy as np
import cv2
import torch

def load_dataset_indexed(cfg, is_train):
    """
    Loading image paths from corresponding files.
    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.
    Returns:
        dataset_cache (list[Dicts]): Indexed dataset cache padded to maxlength
            where a single dict looks like -
            {
                'split': 'train' / 'val',
                'video': <video_id>,
                'frame': <keyframe_num>,
                'max_frames': Maxframe_count,
                'captions': List of dicts where each dict denotes a single bounding box
                            [
                                {
                                    'bbox': [56.80346, 7.09752,165.4739, 268.65045], # [x1, y1, x2, y2]
                                    'caps': [
                                                [3, 73, 4, 65, 81, 81, 81, 81, 81, 81, 81, 81]
                                                [3, 73, 4, 41, 65, 81, 81, 81, 81, 81, 81, 81]
                                                ... other captions for this bounding box
                                            ],
                                    'type': <object_type>
                                }
                                {
                                    ... same for the second bounding box
                                }
                                ...

                            ]
            }
    """

    dataset_cache = json.load(open(
            cfg.DATASET.TRAIN_INDEXED if is_train else cfg.DATASET.VAL_INDEXED
    ))
    print("Loaded indexed dataset")

    return dataset_cache


def get_keyframe_data(boxes_and_labels):
    """
    Getting keyframe indices, boxes and labels in the dataset.
    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.
    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec):
        """
        Convert time index (in second) to frame index.
        0: 900
        30: 901
        """
        return ('ss',1)

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        sec_idx = 0
        keyframe_boxes_and_labels.append([])
        for sec in boxes_and_labels[video_idx].keys():
            if len(boxes_and_labels[video_idx][sec]) > 0:
                keyframe_indices.append(
                    (video_idx, sec_idx, sec, sec)
                )
                keyframe_boxes_and_labels[video_idx].append(
                    boxes_and_labels[video_idx][sec]
                )
                sec_idx += 1
                count += 1
    print("%d keyframes used." % count)

    return keyframe_indices, keyframe_boxes_and_labels


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



def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.
    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.
    Returns:
        imgs (list): list of loaded images. 
    """

    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with PathManager.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))

