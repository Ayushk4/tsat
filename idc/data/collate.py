import torch

def collate_fn(batch):
    frames = [single[0] for single in batch]
    bboxes = [single[1] for single in batch]
    caps = [single[2] for single in batch]
    cap_text = [single[3] for single in batch]

    return frames, torch.tensor([0]), {'bboxes': bboxes,
                                        'caps': caps,
                                        'cap_text': cap_text
                                    }
