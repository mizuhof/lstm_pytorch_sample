import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from utils import AverageMeter


def train(train_loader, model, criterion, optimizer, epoch, args, tf_writer):
    losses = AverageMeter()

    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tf_writer.add_scalars(
            'loss', {'loss_train': losses.val}, epoch*len(train_loader) + i)

    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    print(f'[{epoch}/{args.epochs}]    loss: {losses.avg}')
