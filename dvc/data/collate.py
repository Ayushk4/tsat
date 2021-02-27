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

    keyframes_ixs = torch.tensor(keyframes_ixs)

    return frames, keyframes_ixs, {'bboxes': bboxes, 'caps': caps, 'object_labels': object_labels}
