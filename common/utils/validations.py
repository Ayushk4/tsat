import torch
from torch import nn
import numpy as np

tasks_supported = ["Classification"]

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



def do_validation(config, net, val_loader):
    net.eval()
    correct_instances = 0
    total_instances = 0
    task_type = config.TASK_TYPE
    assert task_type in metrics_supported

    with torch.no_grad():
        for nbatch, batch in enumerate(val_loader):
            images, labels = to_cuda(batch)

            # Forward pass
            outputs = net(images)

            if task_type == "Classification":
                # calculate the accuracy
                predicted = torch.argmax(outputs.data, 1)
                total_instances += labels.size(0)
                correct_instances += (predicted == labels).sum().item()

        if task_type == "Classification":
            val_metrics = (100.0 * correct_instances) / total_instances

        return val_acc
