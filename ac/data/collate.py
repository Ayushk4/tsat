import torch

C_, H_, W_ = 3, 224, 224
pad_image = torch.zeros(1, C_, H_, W_)
padding_fn = lambda x, n: torch.cat([x, pad_image.expand(n, C_, H_, W_)])

def collate_fn(batch):
    labels = torch.LongTensor([single[1] for single in batch])
    frames = [single[0] for single in batch]
    num_frames = [f.shape[0] for f in frames]
    keyframes_ixs = torch.LongTensor([n//2 for n in num_frames])

    max_frames = max(num_frames)
    pad_masks = torch.BoolTensor([[False] * n + [True] * (max_frames - n) for n in num_frames])
    frames = [padding_fn(f, max_frames - n) for f,n in zip(frames, num_frames)]

    frames = torch.stack(frames)
    return frames, keyframes_ixs, pad_masks, labels
