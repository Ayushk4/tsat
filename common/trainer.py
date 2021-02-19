import os
import time
from collections import namedtuple
import torch
import wandb
from prettytable import PrettyTable
from .utils.validations import do_validation
from .utils.losses import calculate_loss_and_accuracy

try:
    from apex import amp
    from apex.amp import _amp_state
except ImportError:
    pass
    # raise ImportError("Please install apex from https://www.github.com/nvidia/apex if you want to use fp16.")

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'rank',
                            'add_step',
                            'data_in_time',
                            'data_transfer_time',
                            'forward_time',
                            'backward_time',
                            'optimizer_time',
                            'metric_time',
                            'eval_metric',
                            'locals'])


def _multiple_callbacks(callbacks, *args, **kwargs):
    """Sends args and kwargs to any configured callbacks.
    This handles the cases where the 'callbacks' variable
    is ``None``, a single function, or a list.
    """
    if isinstance(callbacks, list):
        for cb in callbacks:
            cb(*args, **kwargs)
        return
    if callbacks:
        callbacks(*args, **kwargs)


def to_cuda(batch):
    batch = list(batch)

    for i in range(len(batch)):
        if isinstance(batch[i], torch.Tensor):
            batch[i] = batch[i].cuda(non_blocking=True)
        elif isinstance(batch[i], list):
            for j, o in enumerate(batch[i]):
                if isinstance(batch[i], torch.Tensor):
                    batch[i][j] = o.cuda(non_blocking=True)
        elif isinstance(batch[i], dict):
            for k, v in batch[i].items():
                if isinstance(batch[i][k], torch.Tensor):
                    batch[i][k] = v.cuda(non_blocking=True)

    return batch

def train(config,
          net,
          optimizer,
          train_loader,
          train_metrics,
          val_loader,
          val_metrics,
          criterion,
          rank,
          batch_end_callbacks,
          epoch_end_callbacks,
          use_wandb
        ):

    assert type(use_wandb) == bool
    # TODO
    # fp16=config.TRAIN.fp16 
    # clip_grad_norm = config.TRAIN.CLIP_GRAD_NORM
    # TODO: Add Gradient
    # gradient_accumulate_steps = config.TRAIN.GRADIENT_ACCUMULATE_STEPS
    # assert isinstance(gradient_accumulate_steps, int) and gradient_accumulate_steps >= 1

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        if rank == 0:
            print('PROGRESS: %.2f%%' % (100.0 * epoch / config.TRAIN.END_EPOCH))

        # set epoch as random seed of sampler while distributed training
        # TODO:
        # if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
        #     train_sampler.set_epoch(epoch)

        # reset metrics
        train_metrics.reset()

        # set net to train mode
        net.train()
        criterion.train(0)

        # init end time
        end_time = time.time()

        # training
        for nbatch, batch in enumerate(train_loader):
            # transfer data to GPU
            frames, keyframes_idx, frame_pad_masks, targets = to_cuda(batch)

            outputs = net(frames, keyframes_idx, frame_pad_masks)
            #loss, accuracy = calculate_loss_and_accuracy(config.TASK_TYPE, outputs, labels)
            metric_values = criterion(outputs, targets)

            # store the obtained metrics
            for k, v in metric_values.items():
                train_metrics.store(f'training_{k}', v[0], v[1])

            # clear the gradients
            optimizer.zero_grad()

            # backward time
            loss.backward()

            # optimizer time
            optimizer.step()

            # execute batch_end_callbacks
            # if batch_end_callbacks is not None:
            #     batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, add_step=True, rank=rank)
            #     _multiple_callbacks(batch_end_callbacks, batch_end_params)


            if nbatch % 100 == 0 and rank == 0:
                # Print accuracy and loss
                print('\n---------------------------------')
                print(f'[Rank: {rank if rank is not None else 0}], [Epoch: {epoch}/{config.TRAIN.END_EPOCH}], [Batch: {nbatch}/{len(train_loader)}]')
                print('-----------------------------------\n')

                table = PrettyTable(['Metric', 'Value'])
                for metric_name, metric in train_metrics.all_metrics.items():
                    table.add_row([metric_name, metric.current_value])
                print(table)

        # update end time
        end_time = time.time() - end_time
        if rank == 0:
            print(f'Epoch {epoch} finished in {end_time}s!!')

        # update validation metrics
        val_acc = do_validation(config, net, val_loader)
        val_metrics.store('val_accuracy', val_acc, 'Accuracy')


        # Log both the training and validation metrics
        if rank == 0:
            train_metrics.wandb_log(epoch, use_wandb)
            val_metrics.wandb_log(epoch, use_wandb)
            if use_wandb:
                wandb.log({'Epoch Time':end_time}, step=epoch)

        # print the validation accuracy
        if rank == 0:
            print('\n-----------------')
            print('Validation Metrics')
            print('-----------------\n')

            table = PrettyTable(['Metric', 'Current Value', 'Best Value'])
            for metric_name, metric in val_metrics.all_metrics.items():
                table.add_row([metric_name, metric.current_value, metric.best_value if 'best_value' in dir(metric) else '----'])
            print(table)

        if epoch_end_callbacks is not None:
            _multiple_callbacks(epoch_end_callbacks, rank=rank if rank is not None else 0,
                                epoch=epoch, net=net, optimizer=optimizer
                            )
