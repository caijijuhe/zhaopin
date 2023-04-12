#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   config.py
@Time    :   2022/02/15 13:25:57
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
        parser.add_argument('--epochs', default=100, help='模型训练迭代数')
        parser.add_argument('--use', default='A', help='语料A或B')
        parser.add_argument('--accumulation_steps', default=1, help='梯度累积步数')
        parser.add_argument('--batch_size', default=16, help='训练样本批次数量')
        parser.add_argument('--weight_decay', default=0.001, help='权重衰减值')
        parser.add_argument('--embedding_dim', default=16, help='模型embedding层维度')
        parser.add_argument('--hidden_dim', default=32, help='模型隐藏层维度')
        parser.add_argument('--lr', default=2e-3, help='模型层的学习率')
        parser.add_argument('--warmup_proportion', default=0.1, help='warmup步长占比')
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        # 语料参数
        parser.add_argument('--corpus_dir_train_A', default='BIO/trainBIO/modelA.txt', help='A语料训练文件目录')
        parser.add_argument('--corpus_dir_test_A', default='BIO/testBIO/modelA.txt', help='A语料测试文件目录')
        parser.add_argument('--corpus_dir_train_B', default='BIO/trainBIO/modelB.txt', help='B语料训练文件目录')
        parser.add_argument('--corpus_dir_test_B', default='BIO/testBIO/modelB.txt', help='B语料测试文件目录')
        parser.add_argument('--corpus_dir_tag_A', default='BIO/tags_A.txt', help='A实体标签字典')
        parser.add_argument('--corpus_dir_tag_B', default='BIO/tags_B.txt', help='B实体标签字典')

        parser.add_argument('--entity_collect_file_A', default='entity_pi_A.dat', help='A语料实体数量统计文件')
        parser.add_argument('--entity_collect_file_B', default='entity_pi_B.dat', help='A语料实体数量统计文件')
        # 模型数据保存与加载参数
        parser.add_argument('--load_model_A', default='savedmodel_A/', help='A模型存盘文件')
        parser.add_argument('--load_model_B', default='savedmodel_B/', help='B模型存盘文件')
        parser.add_argument('--load_modelfile', default='', help='模型读取文件')

        return parser

    @staticmethod
    def extension(args):
        # 扩展参数
        options = Dict(args.__dict__)
        # 训练设备
        options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 实体标签
        # A模型标签
        options.categories = {'job': 0, 'major': 1, 'extent': 2, 'wealthfare': 3}
        options.categories_rev = {v: k for k, v in options.categories.items()}
        options.categories_size = len(options.categories)
        # B模型标签
        # options.categories = {'knowledge':0, 'skill':1, 'quality':2}
        # options.categories_rev = {v:k for k,v in options.categories.items()}
        # options.categories_size = len(options.categories)
        return options

    def get_parser(self):
        # 初始化参数解析器
        parser = self.parse()
        # 初始化参数
        parser = self.initialize(parser)
        # 解析命令行参数
        args = parser.parse_args()
        # 扩展参数
        options = self.extension(args)
        return options


def main():
    options = ArgsParse().get_parser()
    for opt in options:
        print(opt, ":", options[opt])
    return options


if __name__ == '__main__':
    main()
