#----------------------------------------
#--------- OS related imports -----------
#----------------------------------------
import os
import wandb

#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as distributed
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

#----------------------------------------
#--------- Model related imports --------
#----------------------------------------
from ac.modules.model import VideoTransformer


#----------------------------------------
#--------- Dataloader related imports ---
#----------------------------------------
from ac.data.build import make_dataloader

#----------------------------------------
#--------- Imports from common ----------
#----------------------------------------
from common.utils.optim import *
from common.utils.load import smart_model_load
from common.utils.misc import summary_parameters
from common.trainer import train
from common.metrics.train_metrics import TrainMetrics
from common.metrics.val_metrics import ValMetrics
from common.callbacks.batch_end_callbacks.speedometer import Speedometer
from common.callbacks.epoch_end_callbacks.checkpoint import Checkpoint

def train_net(args, config):

    # manually set random seed
    if config.RNG_SEED > -1:
        np.random.seed(config.RNG_SEED)
        torch.manual_seed(config.RNG_SEED)
        torch.random.manual_seed(config.RNG_SEED)
        torch.cuda.manual_seed_all(config.RNG_SEED)
        random.seed(config.RNG_SEED)
        torch.backends.cudnn.benchmark = False

    # cudnn
    torch.backends.cudnn.benchmark = False
    if args.cudnn_off:
        torch.backends.cudnn.enabled = False

    # parallel: distributed training for utilising multiple GPUs
    if args.dist:
        # set up the environment
        local_rank = int(os.environ.get('LOCAL_RANK') or 0)
        config.GPUS = str(local_rank)
        torch.cuda.set_device(local_rank)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'] or 23456)
        world_size = int(os.environ['WORLD_SIZE'] or 1)
        rank = int(os.environ['RANK'] or 0)

        # initialize process group
        distributed.init_process_group(
            backend='nccl',
            init_method='tcp://{}:{}'.format(master_address, master_port),
            world_size=world_size,
            rank=rank,
            group_name='mtorch'
        )
        print(f'native distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')

        # set cuda devices
        torch.cuda.set_device(local_rank)
        config.GPUS = str(local_rank)

        # initialize the model and put it to GPU
        model = eval(config.MODEL)(config=config)
        model = model.cuda()

        # wrap the model using torch distributed data parallel
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        # summarize the model
        if rank == 0:
            print("summarizing the main network")
            summary_parameters(model)

        # dataloaders for training and test set
        if config.DATASET.TOY:
            train_loader = make_dataloader(config, mode='train', distributed=True,
                                            num_replicas=world_size, rank=rank)
            val_loader = make_dataloader(config, mode='train', distributed=True,
                                            num_replicas=world_size, rank=rank)

            assert train_loader.dataset.dataset == val_loader.dataset.dataset
        else:
            train_loader = make_dataloader(config, mode='train', distributed=True,
                                            num_replicas=world_size, rank=rank)
            val_loader = make_dataloader(config, mode='val', distributed=True,
                                            num_replicas=world_size, rank=rank)

    else:
        # set CUDA device in env variables
        config.GPUS = [*range(len((config.GPUS).split(',')))] if args.data_parallel else str(0)
        print(f"config.GPUS = {config.GPUS}")

        # initialize the model and put is to GPU
        model = eval(config.MODEL)(config=config)

        if args.data_parallel:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=config.GPUS)

        else:
            torch.cuda.set_device(0)
            model = model.cuda()

        # summarize the model
        print("summarizing the model")
        summary_parameters(model)

        # dataloaders for training and test set
        if config.DATASET.TOY:
            train_loader = make_dataloader(config, mode='train', distributed=False)
            val_loader = make_dataloader(config, mode='train', distributed=False)
        else:
            train_loader = make_dataloader(config, mode='train', distributed=False)
            val_loader = make_dataloader(config, mode='val', distributed=False)

    # set up the initial learning rate
    initial_lr = config.TRAIN.LR

    # configure the optimizer
    try:
        optimizer = eval(f'optim_{config.TRAIN.OPTIMIZER}')(model=model, initial_lr=initial_lr,
                                                            momentum=config.TRAIN.MOMENTUM,
                                                            weight_decay=config.TRAIN.WEIGHT_DECAY)
    except:
        raise ValueError(f'{config.TRAIN.OPTIMIZER}, not supported!!')

    # Load pre-trained model
    if config.NETWORK.PRETRAINED_MODEL != '':
        print(f"Loading the pretrained model from {config.NETWORK.PRETRAINED_MODEL} ...")
        pretrain_state_dict = torch.load(config.NETWORK.PRETRAINED_MODEL,
                                        map_location = lambda storage, loc: storage)['net_state_dict']
        smart_model_load(model, pretrain_state_dict,
                        loading_method=config.NETWORK.PRETRAINED_LOADING_METHOD)

    # Set up the metrics
    train_metrics = TrainMetrics(config, allreduce=args.dist)
    val_metrics = ValMetrics(config, allreduce=args.dist)

    # Set up the callbacks
    # batch end callbacks
    if args.dist:
        batch_end_callbacks = [Speedometer(config.TRAIN.BATCH_SIZE / world_size )]
    else:
        batch_end_callbacks = [Speedometer(config.TRAIN.BATCH_SIZE)]

    # epoch end callbacks
    epoch_end_callbacks = [Checkpoint(config, val_metrics)]

    if args.wandb:
        wandb.watch(model, log='all')

    # At last call the training function from trainer
    train(config=config,
        net=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_metrics=train_metrics,
        val_loader=val_loader,
        val_metrics=val_metrics,
        rank=rank if args.dist else None,
        batch_end_callbacks=batch_end_callbacks,
        epoch_end_callbacks=epoch_end_callbacks,
        use_wandb=args.wandb
        )
