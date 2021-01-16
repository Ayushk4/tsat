import os
import time
from collections import namedtuple
import torch
import wandb
from prettytable import PrettyTable
from utils.validations import do_validation
from utils.losses import calculate_loss_and_accuracy

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

    return batch

def train(config,
          net,
          optimizer,
          train_loader,
          train_metrics,
          val_loader,
          val_metrics,
          rank,
          batch_end_callbacks,
          epoch_end_callbacks,
        ):

    # TODO
    # fp16=config.TRAIN.fp16 
    clip_grad_norm = config.TRAIN.CLIP_GRAD_NORM
    # TODO: Add Gradient
    # gradient_accumulate_steps = config.TRAIN.GRADIENT_ACCUMULATE_STEPS
    # assert isinstance(gradient_accumulate_steps, int) and gradient_accumulate_steps >= 1

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        print('PROGRESS: %.2f%%' % (100.0 * epoch / config.TRAIN.END_EPOCH))

        # set epoch as random seed of sampler while distributed training
        # TODO:
        # if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
        #     train_sampler.set_epoch(epoch)

        # reset metrics
        train_metrics.reset()

        # set net to train mode
        net.train()

        # init end time
        end_time = time.time()

        # training
        for nbatch, batch in enumerate(train_loader):
            global_steps = len(train_loader) * epoch + nbatch

            # transfer data to GPU
            images, labels = to_cuda(batch)

            outputs = net(images)
            loss, accuracy = calculate_loss_and_accuracy(config.TASK_TYPE, outputs, labels)

            # store the obtained metrics
            train_metrics.store('training_loss', loss.item(), 'Loss')
            train_metrics.store('training_accuracy', accuracy, 'Accuracy')

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


            if nbatch % 100 == 0:
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
        print(f'Epoch {epoch} finished in {end_time}s!!')

        # update validation metrics
        val_acc = do_validation(config, net, val_loader, policy_net=policy_net)
        val_metrics.store('val_accuracy', val_acc, 'Accuracy')

        # Log the optimizer stats -- LR
        for i, param_group in enumerate(optimizer.param_groups):
            wandb.log({f'LR_{i}': param_group['lr']}, step=epoch)

        # Log both the training and validation metrics
        train_metrics.wandb_log(epoch)
        val_metrics.wandb_log(epoch)
        wandb.log({'Epoch Time':end_time}, step=epoch)

        # print the validation accuracy
        print('\n-----------------')
        print('Validation Metrics')
        print('-----------------\n')

        table = PrettyTable(['Metric', 'Current Value', 'Best Value'])
        for metric_name, metric in val_metrics.all_metrics.items():
            table.add_row([metric_name, metric.current_value, metric.best_value if 'best_value' in dir(metric) else '----'])
        print(table)

        if epoch_end_callbacks is not None:
            _multiple_callbacks(epoch_end_callbacks, rank=rank if rank is not None else 0, epoch=epoch, net=net, optimizer=optimizer, policy_net=policy_net, policy_optimizer=policy_optimizer, policy_decisions=policy_decisions, policy_max=policy_max, training_strategy=config.NETWORK.TRAINING_STRATEGY)