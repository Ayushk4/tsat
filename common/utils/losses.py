import torch
import torch.nn as nn
import torch.nn.functional as F

task_to_loss = {"Classification": nn.nn.CrossEntropyLoss()}

def calculate_loss_and_accuracy(criterion, outputs, labels):

    loss = criterion(outputs, labels)

    # calculate the accuracy here
    predicted = torch.argmax(outputs.data, 1)
    correct_instances = (predicted == labels).sum().item() 
    total_instances = labels.size(0)
    accuracy = 100.0 * (correct_instances / total_instances)

    return loss, accuracy
