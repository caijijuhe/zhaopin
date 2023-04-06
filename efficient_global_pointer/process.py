#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   process.py
@Time    :   2022/02/15 13:25:41
@Author  :   Mr LaoChen
@Version :   1.0
'''

import os
import json
import torch
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from ner_dataset import NerDataset
from transformers import BertTokenizerFast  # RUST C++类似

from config import ArgsParse
import logging as log

log.basicConfig(level=log.INFO)


def load_json_corpus(corpus_dir):
    """
    读取json语料文件并返回json格式列表
    """
    datas = []
    for f in tqdm(os.listdir(corpus_dir)):
        filename = os.path.join(corpus_dir, f)
        d = json.load(open(filename, encoding='utf-8'))
        datas.append(d)
    return datas


def load_bio_corpus(opt):
    """
    读取bio语料文件并返回json格式列表
    """
    corpus_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), opt.corpus_dir))
    load_file = os.path.join(os.path.dirname(__file__), opt.corpus_dir, opt.corpus_load_file)
    if os.path.exists(load_file):
        datas = torch.load(load_file)
        print(f'语料数据加载成功!语料总数:{len(datas)}')
        return datas

    source_file = os.path.join(corpus_dir, 'source_BIO_2014_cropus.txt')
    target_file = os.path.join(corpus_dir, 'target_BIO_2014_cropus.txt')
    slines = open(source_file, encoding='utf-8').read().split('\n')
    tlines = open(target_file, encoding='utf-8').read().split('\n')

    datas = []
    temp1, temp2 = [], []
    for i in tqdm(range(len(slines))):
        s = slines[i].split()
        t = tlines[i].split()
        # 语料文本
        contents = list()
        # 语料实体标签
        keywords = defaultdict(list)  # {"key":[]}
        start, count, type = (-1, 0, '')
        for i, s in enumerate(s):
            contents.append(s)
            flags = t[i].split('_')
            # 非实体类型tag长度为1
            if len(flags) == 1:
                continue
            if flags[0] == 'B':
                if type != '':
                    keywords[type].append([start, start + count])
                    type = ''
                type = flags[1]
                start = i
                count = 0
            else:
                count += 1
        if type:
            keywords[type].append([start, start + count])
        datas.append({'title': ''.join(contents), 'keywords': keywords})
    torch.save(datas, load_file)
    return datas


def get_dataloader(opt, dataset, tokenizer):
    """
    根据dataset创建dataloader并返回
    """

    def collate(batch_data):
        text_data = [data['text'] for data in batch_data]
        ################################[logging begin]##################################
        log.debug('\n' + '\n'.join(text_data))
        #################################[logging end]###################################
        entities_data = [data['keywords'] for data in batch_data]
        # tokenizer编码
        data = tokenizer(text_data, return_offsets_mapping=True, padding=True, return_tensors='pt')
        # 提取token映射字符数量
        mapping = data.pop('offset_mapping')
        mapping = [{i: range(j[0], j[1] + 1) for i, j in enumerate(map) if j[1] - j[0]} for map in mapping]

        max_len = max([len(d) for d in data['input_ids']])

        # 目标label
        labels = torch.zeros((len(batch_data), opt.categories_size, max_len, max_len))
        for i, entities in enumerate(entities_data):
            for e in entities:
                # 空的实体声明直接跳过
                if len(entities[e]) == 0: continue
                for e_range in entities[e]:
                    # 提取每个实体的声明范围
                    start, end = e_range[0], e_range[1]
                    # 判断对应范围内的token_index
                    ts = [k for k, v in mapping[i].items() if start in v]
                    te = [k for k, v in mapping[i].items() if end in v]

                    if len(ts) > 0 and len(te) > 0:
                        # 末尾的token索引需要加1
                        ts = ts[-1]
                        te = te[0]
                        # label对应类别位置添加标注
                        labels[i, opt.categories[e], ts, te] = 1
                        ################################[logging begin]##################################
                        log.debug(f'{e}:{" ".join([c for c in text_data[i][start:end]])}')
                        log.debug(f'{e}:' + "".join(["{:2d} ".format(i) for i in range(start, end)]))
                        log.debug(f'{e}:字符索引[{start}:{end - 1}]')
                        log.debug(f'{e}:{" ".join(data["input_ids"][i][ts:te + 1].numpy().astype(str))}')
                        log.debug(f'{e}:{tokenizer.decode(data["input_ids"][i][ts:te + 1])}')
                        log.debug(f'{e}:token索引[{ts}_{te}]\n')
                        #################################[logging end]###################################
            data['labels'] = labels
        return data

    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate)


def corpus_split(corpus_data, split_rate=0.8):
    """
    按比率拆分语料
    """
    size = int(len(corpus_data) * split_rate)
    data1 = [d for i, d in enumerate(corpus_data) if i <= size]
    data2 = [d for i, d in enumerate(corpus_data) if i > size]
    return data1, data2


def entity_collect(opt, data_loader):
    """
    统计数据集中不同类别实体数量
    """
    load_file = os.path.join(os.path.dirname(__file__), opt.corpus_dir, opt.entity_collect_file)
    if os.path.exists(load_file):
        entity_classes = torch.load(load_file)
        print('加载实体统计文件成功！')
        print(''.join([f'{k} : {v}\n' for k, v in entity_classes.items()]))
        return entity_classes

    entity_classes = defaultdict(int)
    for data in tqdm(data_loader, desc='分析测试集实体数量'):
        for n, c, s, e in zip(*np.where(data['labels'].numpy())):
            entity_classes[opt.categories_rev[c]] += 1
    torch.save(entity_classes, load_file)
    return entity_classes


if __name__ == '__main__':
    opt = ArgsParse().get_parser()
    # tokenizer
    local = os.path.abspath(os.path.join(os.path.dirname(__file__), opt.local_model_dir, opt.bert_model))
    tokenizer = BertTokenizerFast.from_pretrained(local)
    # # 加载json语料
    # corpus_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'corpus/nested_corpus'))
    # corpus_data = load_json_corpus(corpus_dir)
    # 加载BIO语料
    corpus_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), opt.corpus_dir))
    corpus_data = load_bio_corpus(opt)
    # 拆分语料
    data1, _ = corpus_split(corpus_data)

    # 语料数据集
    # dataset = NerDataset(data1)
    dataset = data1

    # 构建dataloader,为测试观察，batch_size设置为2
    opt.batch_size = 2
    dataloader = get_dataloader(opt, dataset, tokenizer)

    for train_data in dataloader:
        pass
        break
