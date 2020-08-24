import time

import matplotlib.pyplot as plt


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


def loss_fn(pred, target):
    return l1_loss(pred, target) + l2_loss(pred, target)


def l1_loss(pred, target):
    return (target - pred).abs().sum(dim=1).sum(dim=1).sum() / (pred.size(0) * pred.size(1) * pred.size(2))


def l2_loss(pred, target):
    return (target - pred).pow(2).sum(dim=1).sum(dim=1).sum() / (pred.size(0) * pred.size(1) * pred.size(2))


def save_pred_tra(inputs, preds, targets, epoch, tf_writer, tag=''):
    for k, (i, p, t) in enumerate(zip(inputs, preds, targets)):
        missing_i = t[i[:, 0] == 0]

        fig = plt.figure()
        plt.plot(p[:, 0], p[:, 1], marker='o', label='pred')
        plt.plot(t[:, 0], t[:, 1], marker='o', label='target')
        plt.plot(missing_i[:, 0], missing_i[:, 1],
                 marker='o', label='missing')
        plt.legend()

        tf_writer.add_figure(f'{tag}/{k}', fig,
                             global_step=epoch,
                             walltime=time.time())
