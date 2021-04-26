# encoding: utf-8
import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict

import torch
import torch.distributed as dist

from .logger import get_logger

logger = get_logger()

def reduce_tensor(tensor, dst=0, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.reduce(tensor, dst, op)
    if dist.get_rank() == dst:
        tensor.div_(world_size)

    return tensor


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)

    return tensor


def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        state_dict = torch.load(model_file, map_location=device)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        elif 'model_state' in state_dict.keys():
            state_dict = state_dict['model_state']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if not is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'backbone' in k:
                name = k
            elif 'encoder' in k:
                name = k.replace('encoder', 'backbone')
            else:
                name = 'backbone.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model


def parse_devices(input_devices):
    if input_devices.endswith('*'):
        devices = list(range(torch.cuda.device_count()))
        return devices

    devices = []
    for d in input_devices.split(','):
        if '-' in d:
            start_device, end_device = d.split('-')[0], d.split('-')[1]
            assert start_device != ''
            assert end_device != ''
            start_device, end_device = int(start_device), int(end_device)
            assert start_device < end_device
            assert end_device < torch.cuda.device_count()
            for sd in range(start_device, end_device + 1):
                devices.append(sd)
        else:
            device = int(d)
            assert device < torch.cuda.device_count()
            devices.append(device)

    logger.info('using devices {}'.format(
        ', '.join([str(d) for d in devices])))

    return devices


def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _dbg_interactive(var, value):
    from IPython import embed
    embed()