# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def save_checkpoint(state, savedir, itr, last_checkpoints=None, num_checkpoints=None):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filename = os.path.join(savedir, 'checkpt-%08d.pth' % itr)
    torch.save(state, filename)

    if last_checkpoints is not None and num_checkpoints is not None:
        last_checkpoints.append(itr)
        if len(last_checkpoints) > num_checkpoints:
            rm_itr = last_checkpoints.pop(0)
            old_checkpt = os.path.join(savedir, 'checkpt-%08d.pth' % rm_itr)
            if os.path.exists(old_checkpt):
                os.remove(old_checkpt)


def find_latest_checkpoint(savedir):
    import glob
    import re

    checkpt_files = glob.glob(os.path.join(savedir, 'checkpt-[0-9]*.pth'))

    if not checkpt_files:
        return None

    def extract_itr(f):
        s = re.findall('(\d+).pth$', f)
        return int(s[0]) if s else -1

    latest_itr = max(checkpt_files, key=extract_itr)
    return latest_itr


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class ExponentialMovingAverage(object):

    def __init__(self, module, decay=0.999):
        """Initializes the model when .apply() is called the first time.
        This is to take into account data-dependent initialization that occurs in the first iteration."""
        self.decay = decay
        self.module_params = {n: p for (n, p) in module.named_parameters()}
        self.ema_params = {n: p.data.clone() for (n, p) in module.named_parameters()}
        self.nparams = sum(p.numel() for (_, p) in self.ema_params.items())

    def apply(self, decay=None):
        decay = decay or self.decay
        with torch.no_grad():
            for name, param in self.module_params.items():
                self.ema_params[name] -= (1 - decay) * (self.ema_params[name] - param.data)

    def set(self, named_params):
        with torch.no_grad():
            for name, param in named_params.items():
                self.ema_params[name].copy_(param)

    def replace_with_ema(self):
        for name, param in self.module_params.items():
            param.data.copy_(self.ema_params[name])

    def swap(self):
        for name, param in self.module_params.items():
            tmp = self.ema_params[name].clone()
            self.ema_params[name].copy_(param.data)
            param.data.copy_(tmp)

    def __repr__(self):
        return (
            '{}(decay={}, module={}, nparams={})'.format(
                self.__class__.__name__, self.decay, self.module.__class__.__name__, self.nparams
            )
        )
