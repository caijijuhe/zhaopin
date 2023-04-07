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

    # 静态方法 
    @staticmethod
    def parse():
        # 命令行参数解析器
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # 模型训练参数
        parser.add_argument('--epochs', default=50, help='模型训练迭代数')
        parser.add_argument('--learn_rate', default=1e-5, help='学习率')
        parser.add_argument('--batch_size', default=4, help='训练样本批次数量')
        parser.add_argument('--accumulation_steps', default=4, help='梯度累积步数')
        # 模型参数
        parser.add_argument('--bert_model', default='bert-base-chinese', help='bert模型名称')
        parser.add_argument('--local_model_dir', default='bert_model/', help='bert模型本地缓存目录')
        parser.add_argument('--max_position_length', default=1024, help='模型输入position最大长度')
        # 语料参数
        parser.add_argument('--train_file', default='corpus/trainbio4modelA.txt', help='训练语料文件1')
        parser.add_argument('--dev_file', default='corpus/example.dev', help='训练语料文件2')
        parser.add_argument('--test_file', default='corpus/testbio4modelA.txt', help='测试语料文件')
        # 模型保存参数
        parser.add_argument('--save_model_dir', default='saved_model/', help='模型存盘文件夹')
        parser.add_argument('--load_model', default='', help='加载的模型存盘文件')

        return parser

    @staticmethod
    def extension(args):
        # 提取命令行参数通过addict添加其它扩展参数
        # https://github.com/mewwts/addict
        options = Dict(args.__dict__)
        # 训练设备
        options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 实体标签
        options.tags = Dict({
            "O": 0,
            "B-work": 1,
            "I-work": 2,
            "B-extent": 3,
            "I-extent": 4,
            "B-wealthfare": 5,
            "I-wealthfare": 6,
            "B-major": 7,
            "I-major": 8,
            "START": 9,
            "END": 10})
        # 实体标签反向索引    
        options.tags_rev = Dict({v: k for k, v in options.tags.items()})
        # 实体首尾类别对
        options.entity_pair_ix = {
            'work': (1, 2),
            'extent': (3, 4),
            'wealthfare': (5, 6),
            'major': (7, 8)
        }
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
    options = ArgsParse.get_parser()
    for opt in options:
        print(opt, ":", options[opt])
    return options


if __name__ == '__main__':
    main()
