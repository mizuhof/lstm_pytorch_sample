import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from utils import AverageMeter, save_pred_tra


def validate(val_loader, model, criterion, epoch, args, tf_writer=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (inputs_data, targets_data) in enumerate(val_loader):
            inputs = inputs_data.to(args.device)
            targets = targets_data.to(args.device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))

            tf_writer.add_scalars('loss', {'loss_test': losses.avg}, epoch)

    print(f'[validate]    loss: {losses.avg}')
    # save last prediction trajestory
    save_pred_tra(inputs, outputs, targets, epoch, tf_writer)



