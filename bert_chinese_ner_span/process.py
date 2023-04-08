#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   process.py
@Time    :   2022/01/16 09:01:11
@Author  :   Mr LaoChen
@Version :   1.0
'''

from lib2to3.pgen2 import token
import torch
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from config import ArgsParse


def generate_dataloader(dataset, tokenizer, tag_dict, batch_size=32):
    """
    创建模型训练用dataloader并返回
    """

    def collate_fn(batch):
        """
        批次数据转换为模型训练用张量
        """
        tokens, batch_start, batch_end = [], [], []

        for d in batch:
            text = d['text']
            labels = d['labels']

            token = ' '.join([c for c in text])

            tokens.append(token)
            start_tag = torch.zeros((len(text), 1))
            end_tag = torch.zeros((len(text), 1))

            for tag, val in tag_dict.items():
                if len(labels[tag]) > 0:
                    for label_part in labels[tag]:
                        start, end = label_part
                        # if start < 510 and end < 510:
                        # 起始位置采用奇数标记
                        start_tag[start] = val * 2 - 1
                        # 结束位置采用偶数标记
                        end_tag[end] = val * 2
            # 针对【方案1】label
            batch_start.append(start_tag)
            batch_end.append(end_tag)

            # 针对【方案2】label
            # batch_start.append([0] + start_tag + [0])
            # batch_end.append([0] + end_tag + [0])

        train_data = tokenizer(tokens, padding=True, return_tensors='pt', max_length=1024, truncation=True)
        batch_start = pad_sequence(batch_start, batch_first=True)
        batch_end = pad_sequence(batch_end, batch_first=True)
        # label也存入train_data，与model的输入参数对应
        train_data['start_ids'] = batch_start.long()
        train_data['end_ids'] = batch_end.long()

        return train_data

    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn)


def entity_collect(opt, dataset):
    # 实体计数用
    entity_count = {k: 0 for k in opt.tags}
    for json_data in dataset:
        labels = json_data['labels']
        for k in opt.tags:
            entity_count[k] += len(labels[k])
    return entity_count


if __name__ == '__main__':
    import os
    from transformers import AutoTokenizer
    from ner_dataset import NerIterableDataset
    from config import ArgsParse

    opt = ArgsParse.get_parser()

    train_file = os.path.join(os.path.dirname(__file__), opt.train_file)
    dataset = NerIterableDataset(train_file)
    tokenizer = AutoTokenizer.from_pretrained(opt.bert_model)

    dataloader = generate_dataloader(dataset, tokenizer, opt.tags, 4)

    for _data in dataloader:
        input_ids = _data['input_ids']
        token_type_ids = _data['token_type_ids']
        attention_mask = _data['attention_mask']
        start = _data['start_ids']
        end = _data['end_ids']

        print(input_ids.shape)
        print(token_type_ids.shape)
        print(attention_mask.shape)

        print(start.shape)
        print(end.shape)

        print(start.reshape(-1))
        print(end.reshape(-1))
        break
