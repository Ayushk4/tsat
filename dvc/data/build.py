#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch

#----------------------------------------
#--------- Funcs and Classes for Datasets
#----------------------------------------
from datasets.stdc import STDC

DATASET_CATALOGS = {'stdc':STDC}

def build_dataset(dataset_name, *args, **kwargs):
    assert dataset_name in DATASET_CATALOGS, "dataset not in catalogs"
    return DATASET_CATALOGS[dataset_name](*args, **kwargs)

def make_data_sampler(dataset, shuffle, distributed, num_replicas, rank):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle,
                                                            num_replicas=num_replicas, rank=rank)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def make_batch_data_sampler(dataset, sampler, batch_size):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)
    return batch_sampler

def make_dataloader(config, dataset=None, mode='train', distributed=False, num_replicas=None, rank=None):

    # config variables
    num_gpu = len(config.GPUS) if isinstance(config.GPUS, list) else len(config.GPUS.split(','))
    num_workers = config.NUM_WORKERS_PER_GPU * num_gpu
    num_replicas = 1 if num_replicas is None else num_replicas

    if mode == 'train':
        batch_size = int(config.TRAIN.BATCH_SIZE / num_replicas)
        shuffle = config.TRAIN.SHUFFLE
        splits = config.DATASET.TRAIN_SPLIT
    else:
        batch_size = int(config.VAL.BATCH_SIZE / num_replicas)
        shuffle = config.VAL.SHUFFLE
        splits = config.DATASET.VAL_SPLIT

    # create a Dataset class object
    if dataset is None:
        dataset = build_dataset(config=config,
                                split=split
                            )

    sampler = make_data_sampler(dataset, shuffle, distributed, num_replicas, rank)
    batch_sampler = make_batch_data_sampler(dataset, sampler, batch_size)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             pin_memory=False)

    return dataloader
