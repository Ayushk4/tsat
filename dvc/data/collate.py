import torch

box_pad = torch.zeros(1, 4)
zero_longtensor_pad = torch.LongTensor([0])
box_padding_fn = lambda x, n: torch.cat([x, box_pad.expand(n, 4)])
caps_padding_fn = lambda x, n: torch.cat([x, zero_longtensor_pad.unsqueeze(-1).expand(n, x.shape[-1])])
obj_padding_fn = lambda x, n: torch.cat([x, zero_longtensor_pad.expand(n)])


def collate_fn(batch):
    frames = [single[0] for single in batch]
    bboxes = [torch.tensor(single[1]) for single in batch]
    caps = [torch.LongTensor(single[2]) for single in batch]
    object_labels = [torch.LongTensor(single[3]) for single in batch]
    keyframes_ixs = [single[4] for single in batch]
    # pad_captions = batch[0][-1]
    # num_boxes = [f.shape[0] for f in bboxes]

    # max_boxes = max(num_boxes)
    # boxes_pad_masks = torch.BoolTensor([[False] * n + [True] * (max_boxes - n) for n in num_boxes])

    # bboxes = [box_padding_fn(f, max_boxes - n) for f,n in zip(bboxes, num_boxes)]
    # if pad_captions:
        # caps = [caps_padding_fn(f, max_boxes - n) for f,n in zip(caps, num_boxes)]
    # object_labels = [obj_padding_fn(f, max_boxes - n) for f,n in zip(object_labels, num_boxes)]

    # frames = torch.stack(frames)
    # bboxes = torch.stack(bboxes)
    # if pad_captions:
        # caps = torch.stack(caps)
    keyframes_ixs = torch.tensor(keyframes_ixs)

    return frames, keyframes_ixs, {'boxes_pad_masks': boxes_pad_masks, 'bboxes': bboxes, 'caps': caps, 'object_labels': object_labels}
