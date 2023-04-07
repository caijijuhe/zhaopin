#!/usr/bin/env python
# -*- coding:utf-8
'''
@File	:	corpus_proc.py
@Time	:	2022/01/01 09:14:56
@Author	:	Mr LaoChen
@Version:	1.0
'''

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from config import ArgsParse
from ner_dataset import NerDataset
from torch.nn.utils.rnn import pad_sequence

from model_utils import custom_local_bert_tokenizer

def read_corpus(corpu_files):
    """
    读取语料文件中的字符、标记并返回
    """
    sentences, sent_tags = [],[]
    s,t = [],[]
    # for cf in corpu_files:
    # for line in open(os.path.join(os.path.dirname(__file__),cf), encoding='UTF-8'):
    for line in open(corpu_files, 'r', encoding='UTF-8'):
        line = line.strip()
        if line == '':
            # if len(s) > 510:
            #     continue
            sentences.append(' '.join(s))
            sent_tags.append(' '.join(t))
            s,t = [],[]
            continue
        ss,tt = line.split()
        s.append(ss)
        t.append(tt)
    return sentences, sent_tags

def generate_dataloader(dataset,tokenizer,tag_dict,batch_size):
    """
    创建模型训练用dataloader并返回
    """
    def collate_fn(batch):
        """
        批次数据转换为模型训练用张量
        """
        batch = np.array(batch)
        batch_data = tokenizer(list(batch[:,0]),padding=True,return_tensors='pt')
        label_list,mask_list = [],[]
        for tag in batch[:,1]:
            # label index
            label = torch.tensor([tag_dict['START']] + [tag_dict[t] for t in tag.split()] + [tag_dict['END']])
            label_list.append(label)
            # label mask
            mask = torch.ones_like(label).bool()
            mask_list.append(mask)
        # padding by longets
        label_list = pad_sequence(label_list, batch_first = True)
        mask_list = pad_sequence(mask_list, batch_first = True, padding_value = False)

        batch_data['label_ids'] = label_list
        batch_data['label_mask'] = mask_list

        return batch_data
        
    return DataLoader(dataset=dataset,batch_size=batch_size,collate_fn=collate_fn)

if __name__ == '__main__':
    from config import ArgsParse
    from model_utils import custom_local_bert

    opt = ArgsParse().get_parser()
    sentences, sent_tags = read_corpus([opt.train_file])

    # 测试Tokenizer
    ckpt = 'bert-base-chinese'
    local = os.path.join(os.path.dirname(__file__), opt.local_model_dir, opt.bert_model)
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(local, model_max_length = opt.max_position_length)

    # 测试dataloader
    dataset = NerDataset(sentences, sent_tags)
    data_loader = generate_dataloader(dataset, tokenizer, opt.tags, 4)
    for data in data_loader:
        print(data)
        break