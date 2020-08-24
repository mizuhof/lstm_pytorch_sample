import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from utils import AverageMeter, save_pred_tra


def validate(val_loader, model, criterion, epoch, args, tf_writer=None):
    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (inputs_data, targets_data) in enumerate(val_loader):
            inputs = inputs_data.to(args.device)
            targets = targets_data.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))

    print(f'[validate]    loss: {losses.avg}')
    tf_writer.add_scalars('loss', {'loss_test': losses.avg}, epoch)

    # save last prediction trajestory
    save_data_num = int(inputs.size(0)) if int(inputs.size(0)) < 100 else 100
    save_pred_tra(inputs[:save_data_num].cpu().numpy(),
                  outputs[:save_data_num].cpu().numpy(),
                  targets[:save_data_num].cpu().numpy(), epoch, tf_writer, tag='val')
