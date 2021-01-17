import _init_paths
import os
import argparse
import torch
import subprocess

from ac.functions.config import config, update_config
from ac.functions.train import train_net
#from ac.functions.test import test_net

import wandb

def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--model-dir', type=str, help='root path to store checkpoint')
    parser.add_argument('--dist', help='whether to use distributed training', default=False, action='store_true')
    # parser.add_argument('--slurm', help='whether this is a slurm job', default=False, action='store_true')
    parser.add_argument('--do-test', help='whether to generate csv result on test set',
                        default=False, action='store_true')
    parser.add_argument('--cudnn-off', help='disable cudnn', default=False, action='store_true')
    parser.add_argument('--wandb', required=True, type=str, help="Enter true/false/yes/no or t/f/y/n")
    parser.add_argument('--run', required=True, type=str)
    args = parser.parse_args()

    assert args.wandb.lower() in ['t', 'f', 'y', 'n', "yes", "no", "true", "false", ]
    if args.wandb[0] in ['t', 'y']:
        args.wandb = True
    elif args.wandb[0] in ['f', 'n']:
        args.wandb = False
    else:
        raise NotImplementedError

    if args.cfg is not None:
        update_config(args.cfg)
    if args.model_dir is not None:
        config.OUTPUT_PATH = config.OUTPUT_PATH

    config.RUN += args.run

    # if args.slurm:
        # proc_id = int(os.environ['SLURM_PROCID'])
        # ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = os.environ['SLURM_NODELIST']
        # num_gpus = torch.cuda.device_count()
        # addr = subprocess.getoutput(
        #     'scontrol show hostname {} | head -n1'.format(node_list))
        # os.environ['MASTER_PORT'] = str(29500)
        # os.environ['MASTER_ADDR'] = addr
        # os.environ['WORLD_SIZE'] = str(ntasks)
        # os.environ['RANK'] = str(proc_id)
        # os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)

    return args, config


def main():
    args, config = parse_args()

    if args.wandb:
        if os.popen('git status --porcelain').read() != "":
            print("\n\n\n==================================================\n\n")
            print("Git commit the current code for reproductibility!")
            print("\n\n\n==================================================\n\n")
            raise "Git Commit Error"

        # initialize wandb
        if not args.dist or rank == 0:
            wandb.init(entity="Ayushk4", project=config.PROJECT, name=config.RUN, config=config)

    train_net(args, config)

    if args.do_test and (rank is None or rank == 0):
        test_net(args, config)


if __name__ == '__main__':
    main()

