import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset
from model import Model
from opts import argparser
from train import train
from utils import l2_loss
from val import validate


def save_checkpoint(state, epoch, args):
    filename = f'{args.checkpoint_dir}/ckpt_{epoch}.pth'
    torch.save(state, filename)


def check_log_folders(args):
    """Create log and checkpoint folder"""
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)


def main():
    args = argparser()
    args.log_dir = os.path.join(args.log_path, 'log')
    args.checkpoint_dir = os.path.join(args.log_path, 'checkpoint')
    args.model_dir = os.path.join(args.log_path, 'model')
    check_log_folders(args)

    model = Model()

    # save model graph visualization
    model_writer = SummaryWriter(os.path.join(args.log_path, 'model'))
    inputs = torch.rand(8, 10, 2)
    model_writer.add_graph(model, inputs)
    model_writer.close()

    if args.device_ids != '' and torch.cuda.device_count() > 1:
        args.device_ids = args.device_ids.split(",")
        args.device_ids = list(map(lambda x: int(x), args.device_ids))
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        print(args.device_ids)
        torch.cuda.set_device(args.device_ids[0])
        model = nn.DataParallel(model, device_ids=args.device_ids).cuda()
    else:
        args.device = torch.device(
            "cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
        model.to(args.device)

    train_dataset = Dataset()
    val_dataset = Dataset()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    optimizer = optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    criterion = l2_loss

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0, last_epoch=-1
    )
    tf_writer = SummaryWriter(log_dir=os.path.join(args.log_path, 'log'))

    for epoch in range(args.epochs):
        # train for one epoch
        train(train_loader, model, criterion,
              optimizer, epoch, args, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            validate(val_loader, model, criterion, epoch *
                     len(train_loader), args, tf_writer)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, args)

        scheduler.step()


if __name__ == '__main__':
    main()
