"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.
Description :
Author：Team Li
"""

import argparse
from utils import print_config, process_config

def get_b16_config(config):
    """ ViT-B/16 configuration """
    config.patch_size = 16
    config.emb_dim = 768
    config.mlp_dim = 3072
    config.num_heads = 12
    config.num_layers = 12
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config

def get_train_config():
    parser = argparse.ArgumentParser("Visual Transformer Train/Fine-tune")

    # basic config
    parser.add_argument("--exp-name", type=str, default="ft", help="experiment name")
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--tensorboard", default=False, action='store_true', help='flag of turnning on tensorboard')
    parser.add_argument("--model-arch", type=str, default="b16", help='model setting to use',
                        choices=['b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=384, help="input image size", choices=[224, 384])
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--train-steps", type=int, default=10000, help="number of training/fine-tunning steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help='weight decay')
    parser.add_argument("--warmup-steps", type=int, default=500, help='learning rate warm up steps')
    parser.add_argument("--data-dir", type=str, default='../data', help='data folder')
    parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for fine-tunning/evaluation")
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
    config = parser.parse_args()

    # model config
    config = eval("get_{}_config".format(config.model_arch))(config)
    process_config(config)
    print_config(config)
    return config

def get_eval_config():
    parser = argparse.ArgumentParser("Visual Transformer Evaluation")

    # basic config
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--model-arch", type=str, default="b16", help='model setting to use',
                        choices=['b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=384, help="input image size", choices=[224, 384])
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--data-dir", type=str, default='./data', help='data folder')
    parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for fine-tunning/evaluation")
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
    config = parser.parse_args()

    # model config
    config = eval("get_{}_config".format(config.model_arch))(config)

    print_config(config)
    return config