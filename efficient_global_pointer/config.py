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
        parser.add_argument('--epochs', default=30, help='模型训练迭代数')
        parser.add_argument('--bert_lr', default=2e-5, help='bert模型的学习率')
        parser.add_argument('--other_lr', default=2e-3, help='非bert模型层的学习率')
        parser.add_argument('--batch_size', default=4, help='训练样本批次数量')
        parser.add_argument('--accumulation_steps', default=6, help='梯度累积步数')
        parser.add_argument('--head_size', default=64, help='位置编码层大小')
        parser.add_argument('--mid_linear_dims', default=128, help='隐藏层大小')
        parser.add_argument('--dropout', default=0.1, help='dropout层比率')
        parser.add_argument('--per_steps_loss', default=100, help='模型训练计算平均loss的间隔')
        parser.add_argument('--use_amp', default=False, help='是否使用混合精度训练')
        parser.add_argument('--warmup_proportion', default=0.1, help='warmup步长占比')
        parser.add_argument('--weight_decay', default=0.001, help='权重衰减值')
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        # 模型参数
        parser.add_argument('--bert_model', default='bert-base-chinese', help='bert模型名称')
        parser.add_argument('--local_model_dir', default='bert_model/', help='bert模型本地缓存目录')
        parser.add_argument('--max_length', default=1500, help='模型输入position最大长度')
        # 语料参数
        parser.add_argument('--corpus_dir', default='corpus/zhaopin', help='语料文件目录')
        parser.add_argument('--corpus_load_file', default='corpus4pointer.json', help='语料数据存盘文件')
        parser.add_argument('--entity_collect_file', default='entity_collect.data', help='语料实体数量统计文件')
        # 模型数据保存参数
        parser.add_argument('--save_model_dir', default='./saved_model/', help='模型存盘文件夹')
        parser.add_argument('--load_model', default='', help='模型存盘文件')

        return parser

    @staticmethod
    def extension(args):
        # 扩展参数
        options = Dict(args.__dict__)
        # 训练设备
        options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 实体标签
        # options.categories = {"标的物": 0, "行政区": 1, "单位": 2, "项目":3, "时间":4, "招标方式":5, "项目编号":6}
        # options.categories_rev = {1:"标的物", 2:"行政区", 3:"单位", 4:"项目", 5:"时间", 6:"招标方式", 7:"项目编号"}
        # options.categories = {'T':0, 'PER':1, 'ORG':2, 'LOC':3}
        # options.categories_rev = {0:'T', 1:'PER', 2:'ORG', 3:'LOC'}
        options.categories = {'job': 0, 'major': 1, 'knowledge': 2, 'quality': 3, 'skill': 4, 'extent': 5,
                              'wealthfare': 6}
        options.categories_rev = {0: 'job', 1: 'major', 2: 'knowledge', 3: 'quality', 4: 'skill', 5: 'extent',
                                  6: 'wealthfare'}
        options.categories_size = len(options.categories)
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
