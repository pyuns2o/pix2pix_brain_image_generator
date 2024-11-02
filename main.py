import argparse
from train import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test', type=str)
parser.add_argument('--data_dir', default="/home/yspark/datasets/dHCP", type=str)
parser.add_argument('--ckpt_dir', default="/home/yspark/pix2pix/checkpoint_2", type=str)
parser.add_argument('--log_dir', default="/home/yspark/pix2pix/log_2", type=str)
parser.add_argument('--result_dir', default="/camin1/yspark//result_2", type=str)

parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--num_epoch', default=300, type=int)

parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--beta1', default=0.5, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--cuda', default='cuda:0', type=str)

args = parser.parse_args()

if args.mode == 'train':
    train(args)
elif args.mode == 'test':
    test(args)
