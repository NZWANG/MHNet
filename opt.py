import datetime
import argparse
import random
import numpy as np
import torch
dataset = 'abide'

class OptInit:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of EV-GCN')
        parser.add_argument('--train', default=1, type=int, help='train(default) or evaluate')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        parser.add_argument('--seed', type=int, default=1, help='random state')
        parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=5e-3, type=float, help='weight decay')
        parser.add_argument('--num_iter', default=240, type=int, help='number of epochs for training')
        parser.add_argument('--folds', default=10, type=int, help='cross validation')
        parser.add_argument('--edropout', type=float, default=0.3, help='edge dropout rate')
        parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')
        parser.add_argument('--log_save', type=bool, default=True, help='save log or not')
        parser.add_argument('--model_save', type=bool, default=True, help='save result or not')
        parser.add_argument('--datapath', type=str, default=r'./test', help='save result or not')
        parser.add_argument('--dataset', default=dataset, type=str, help='name of dataset')
        parser.add_argument('--ckpt_path', type=str, default=rf'./save_model/{dataset}/',
                            help='checkpoint path to save trained models')
        parser.add_argument('--batch_size', type=int, default=24)
        parser.add_argument('--print_freq', default=5, type=int, help='print frequency')
        args = parser.parse_args()

        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device('cpu')
            print(" Using CPU in torch")
        else:
            args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(" Using GPU in torch")

        self.args = args

    def print_args(self):
        # self.args.printer args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train==1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        self.set_seed(123)
        self.print_args()
        return self.args

    def set_seed(self, seed=0):
        """固定系统种子随机值"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


