import csv
import random
from functools import partialmethod

import os
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


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


class Logger(object):

    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets, topk=1):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(topk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def calculate_class_num(targets, label_dict, flag):
    res = 'In ' + flag + ' , '
    class_num = [0]*len(label_dict)

    for each in targets:
        class_num[each] += 1

    for key, value in label_dict.items():
        res += key + ': ' + str(class_num[value]) + ' || '

    print(res)


def calculate_precision_and_recall(outputs, targets, pos_label=1):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        precision, recall, _, _ = precision_recall_fscore_support(
            targets.view(-1, 1).cpu().numpy(),
            pred.cpu().numpy())

        return precision[pos_label], recall[pos_label]


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def partialclass(cls, *args, **kwargs):

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass


def save_checkpoint(checkpoint_path, epoch, model, optimizer, info=None):
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not info:
        info = epoch
    save_file_path = os.path.join(checkpoint_path, 'save_{}.pth'.format(info))
    states = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)