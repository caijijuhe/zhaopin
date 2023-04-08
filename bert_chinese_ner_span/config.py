#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   config.py
@Time    :   2022/01/23 13:05:22
@Author  :   Mr LaoChen
@Version :   1.0
'''

import torch
import argparse
from addict import Dict


class ArgsParse:

    @staticmethod
    def parse():
        # 命令行参数解析器
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # 模型训练参数
        parser.add_argument('--epochs', default=50, help='模型训练迭代数')
        parser.add_argument('--bert_lr', default=4e-5, help='bert模型的学习率')
        parser.add_argument('--other_lr', default=4e-3, help='非bert模型层的学习率')
        parser.add_argument('--batch_size', default=2, help='训练样本批次数量')
        parser.add_argument('--mid_linear_dims', default=128, help='隐藏层大小')
        parser.add_argument('--dropout', default=0.1, help='dropout层比率')
        parser.add_argument('--per_steps_loss', default=100, help='模型训练计算平均loss的间隔')
        parser.add_argument('--use_amp', default=True, help='是否使用混合精度训练')
        parser.add_argument('--warmup_proportion', default=0.1, help='warmup步长占比')
        parser.add_argument('--weight_decay', default=0.001, help='权重衰减值')
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--step_train_loss', default=4)
        # 模型参数
        parser.add_argument('--bert_model', default='bert-base-chinese', help='bert模型名称')
        parser.add_argument('--local_model_dir', default='bert_model/', help='bert模型本地缓存目录')
        parser.add_argument('--max_position_length', default=1024, help='模型输入position最大长度')
        # 语料参数
        parser.add_argument('--train_file', default='corpus/msra_mid/msra_train.json', help='训练语料文件')
        parser.add_argument('--test_file', default='corpus/msra_mid/msra_test.json', help='测试语料文件')
        # 模型保存参数
        parser.add_argument('--save_model_dir', default='saved_model/', help='模型存盘文件夹')
        parser.add_argument('--load_model', default='', help='加载的模型存盘文件')

        return parser

    @staticmethod
    def extension(args):
        # 扩展参数
        options = Dict(args.__dict__)
        # 训练设备
        options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 实体标签
        options.tags = {"job": 1, "major": 2, "extent": 3, "wealthfare":4}
        options.tags_rev = Dict({v: k for k, v in options.tags.items()})

        return options

    @staticmethod
    def get_parser():
        # 初始化参数解析器
        parser = ArgsParse.parse()
        # 初始化参数
        parser = ArgsParse.initialize(parser)
        # 解析命令行参数
        args = parser.parse_args()
        # 扩展参数
        options = ArgsParse.extension(args)
        return options


def main():
    options = ArgsParse().get_parser()
    for opt in options:
        print(opt, ":", options[opt])
    return options


if __name__ == '__main__':
    main()
