#----------------------------------------
#--------- Torch Related Imports --------
#----------------------------------------
import torch
import torch.distributed as distributed

#----------------------------------------
#--------- Import Wandb Here ------------
#----------------------------------------
import wandb


class TrainAccuracyMetric():

    def __init__(self, initial_value, allreduce=False, **kwargs):

        self.current_value = initial_value
        self.iterations = 1
        self.allreduce = allreduce

    def update(self, new_value):

        self.current_value = (self.current_value - (self.current_value-new_value)/(self.iterations + 1))

        # If all reduce, get the number of GPUs
        if self.allreduce:
            gpus = torch.tensor(1.0).cuda()

            # convert to tensor
            cv = torch.tensor(self.current_value).cuda()

            distributed.all_reduce(cv, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(gpus, op=distributed.ReduceOp.SUM)

            self.current_value = cv.item()/gpus.item()

        self.iterations += 1

    def wandb_log(self, metric_name, step):

        wandb.log({metric_name: self.current_value}, step=step)

class TrainLossMetric():

    def __init__(self, initial_value, **kwargs):

        self.current_value = initial_value

    def update(self, new_value):

        self.current_value = new_value

    def wandb_log(self, metric_name, step):

        wandb.log({metric_name: self.current_value}, step=step)

class ValAccuracyMetric():

    def __init__(self, initial_value, allreduce=False, **kwargs):

        self.current_value = initial_value
        self.best_value = initial_value
        self.updated_best_val = True
        self.allreduce = allreduce

    def update(self, new_value):

        self.current_value = new_value

        # If all reduce, get the number of GPUs
        if self.allreduce:
            gpus = torch.tensor(1.0).cuda()

            # convert to tensor
            cv = torch.tensor(self.current_value).cuda()

            distributed.all_reduce(cv, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(gpus, op=distributed.ReduceOp.SUM)

            self.current_value = cv.item()/gpus.item()

        if self.current_value > self.best_value:
            self.best_value = self.current_value
            self.updated_best_val = True
        else:
            self.updated_best_val = False

    def wandb_log(self, metric_name, step):

        wandb.log({f'current_{metric_name}': self.current_value, f'best_{metric_name}': self.best_value}, step=step)
