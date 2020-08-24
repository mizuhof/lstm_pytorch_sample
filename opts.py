import argparse


def argparser():
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of Temporal Segment Networks")

    parser.add_argument('--log_path', type=str, default="log")
    parser.add_argument('--device_ids', type=str, default="")
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='number of evalation frequency')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')

    return parser.parse_args()
