import os
import numpy as np
import yaml
import argparse
from kws.bin.trainer import Trainer
import logging
import torch
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--is_eval',
                        type=bool,
                        default=False,
                        help='If its the evaluation phase?')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=4,
                        type=int,
                        help='num of subprocess workers for reading')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Set random seed
    torch.manual_seed(777)
    print(args)
    distributed = args.world_size > 1
    with open(args.config) as f:
        params = yaml.safe_load(f)
    
    trainer = Trainer(params, args)
    trainer.evaluate()
    
