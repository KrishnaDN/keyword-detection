from __future__ import print_function

import argparse
import copy
import logging
import os
import warnings
import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from kws.dataset.dataset import AudioDataset
from kws.dataset.helpers import collate_fun
from kws.model import Models
from kws.bin import load_checkpoint, save_checkpoint
from kws.bin import BuildOptimizer, BuildScheduler
from kws.bin import Executor
warnings.filterwarnings("ignore")



def initialize_model(params):
    model = Models[params['model']['model_type']](params['model'])
    return model


def Trainer(params, args):
    cmvn_file = params['data']['cmvn_file']
    data_file = params['data']['train']
    labels = params['data']['labels']
    train_dataset = AudioDataset(data_file, cmvn_file, labels, **params['dataset_conf'])
    
    cv_dataset_conf = copy.deepcopy(params['dataset_conf'])
    cv_dataset_conf['spec_augment'] = False
    cv_dataset_conf['spec_substitute'] = False
    cmvn_file = params['data']['cmvn_file']
    data_file = params['data']['valid']
    labels = params['data']['labels']
    cv_dataset = AudioDataset(data_file, cmvn_file, labels, **cv_dataset_conf)
    distributed = args.world_size > 1
    if distributed:
        logging.info('training on multiple gpus, this gpu {}'.format(args.gpu))
        dist.init_process_group(args.dist_backend,
                                init_method=args.init_method,
                                world_size=args.world_size,
                                rank=args.rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
        cv_sampler = torch.utils.data.distributed.DistributedSampler(
            cv_dataset, shuffle=False)
    else:
        train_sampler = None
        cv_sampler = None

    train_data_loader = DataLoader(train_dataset,
                                   collate_fn=collate_fun,
                                   sampler=train_sampler,
                                   shuffle=(train_sampler is None),
                                   batch_size=params['train']['batch_size'],
                                   num_workers=args.num_workers)
    cv_data_loader = DataLoader(cv_dataset,
                                collate_fn=collate_fun,
                                sampler=cv_sampler,
                                shuffle=False,
                                batch_size=params['train']['batch_size'],
                                num_workers=args.num_workers)
    model = initialize_model(params)
    print(model)
    executor = Executor()
    
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    num_epochs = params['train']['epochs']
    model_dir = os.path.join(params['train']['exp_dir'],params['train']['model_dir'])
    writer = None
    if args.rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(model_dir, exp_id))

    if distributed:
        assert (torch.cuda.is_available())
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
        device = torch.device("cuda")
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

    optimizer = optimizer = BuildOptimizer[params['train']['optimizer_type']](
        filter(lambda p: p.requires_grad, model.parameters()), **params['train']['optimizer']
    )
    scheduler = BuildScheduler[params['train']['scheduler_type']](optimizer, **params['train']['scheduler'])
    final_epoch = None
    params['rank'] = args.rank
    params['is_distributed'] = distributed
    if start_epoch == 0 and args.rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)
    
    executor.step = step
    scheduler.step()
    for epoch in range(start_epoch, num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_data_loader, device,
                       writer, params)
        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device,
                                                params)
        if args.world_size > 1:
            # all_reduce expected a sequence parameter, so we use [num_seen_utts].
            num_seen_utts = torch.Tensor([num_seen_utts]).to(device)
            # the default operator in all_reduce function is sum.
            dist.all_reduce(num_seen_utts)
            total_loss = torch.Tensor([total_loss]).to(device)
            dist.all_reduce(total_loss)
            cv_loss = total_loss[0] / num_seen_utts[0]
            cv_loss = cv_loss.item()
        else:
            cv_loss = total_loss / num_seen_utts

        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        if args.rank == 0:
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                model, save_model_path, {
                    'epoch': epoch,
                    'lr': lr,
                    'cv_loss': cv_loss,
                    'step': executor.step
                })
            writer.add_scalars('epoch', {'cv_loss': cv_loss, 'lr': lr}, epoch)
        final_epoch = epoch

    if final_epoch is not None and args.rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
    
    
    

    
