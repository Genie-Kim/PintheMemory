"""
Pytorch Optimizer and Scheduler Related Task
"""
import math
import logging
import torch
from torch import optim
from config import cfg


def get_optimizer(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    base_params = []

    for name, param in net.named_parameters():
        base_params.append(param)

    if args.sgd:
        optimizer = optim.SGD(base_params,
                            lr=args.lr,
                            weight_decay=5e-4, #args.weight_decay,
                            momentum=args.momentum,
                            nesterov=False)
    else:
        raise ValueError('Not a valid optimizer')

    lambda1 = lambda iteration: math.exp(-1 * args.poly_exp * iteration / 120000)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    return optimizer, scheduler


def load_weights(net, optimizer, scheduler, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
    logging.info("Loading weights from model %s", snapshot_file)
    net, optimizer, scheduler, epoch, mean_iu = restore_snapshot(net, optimizer, scheduler, snapshot_file,
            restore_optimizer_bool)
    return epoch, mean_iu


def restore_snapshot(net, optimizer, scheduler, snapshot, restore_optimizer_bool):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logging.info("Checkpoint Loaded")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint and restore_optimizer_bool:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
        logging.info("Checkpoint network state_dict uploaded!!")
    else:
        net = forgiving_state_restore(net, checkpoint)
        logging.info("Checkpoint network uploaded!!")

    if 'memory' in checkpoint:
        try:
            net.module.memory.m_items = checkpoint['memory'].cuda()
            logging.info("Checkpoint memory uploaded!!")
        except AttributeError:
            print("There is no memory in the network!!, Memory in the pretrained model did not uploaded")

    return net, optimizer, scheduler, checkpoint['epoch'], checkpoint['mean_iu']


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            print("Do not match with saved parameter ", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

def forgiving_state_copy(target_net, source_net):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = target_net.state_dict()
    loaded_dict = source_net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
            print("Matched", k)
        else:
            print("Skipped loading parameter ", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    target_net.load_state_dict(net_state_dict)
    return target_net
