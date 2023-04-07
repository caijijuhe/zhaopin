#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bert_bilstm_crf.py
@Time    :   2022/09/07 
@Author  :   Mr LaoChen
@Version :   1.1
'''

import os
import torch
from torch import nn
from torchcrf import CRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertCRF(nn.Module):
    def __init__(self, bert_model, target_size):
        super(BertCRF, self).__init__()
        # bert模型
        self.bert = bert_model
        # Linear
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, target_size)
        # CRF层
        self.crf = CRF(target_size, batch_first=True)
        """
        CRF 目标通过状态转移矩阵，规约正确状态概率
        B-PER: [START->B-PER, END->B-PER, I-PER->B-PER, ....]
        I-PER: [START->I-PER, END->I-PER, ....]
        .....: [....] 
        """

    def loss(self, out, target, mask):
        return -1 * self.crf(out, target, mask)

    def decode(self, out, mask):
        return self.crf.decode(out, mask)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # bert模型输出
        with torch.no_grad():
            hidden_states = self.bert(input_ids, token_type_ids, attention_mask)
        # bert最后一层的hidden_state
        bert_out = hidden_states.last_hidden_state
        # 推理
        features = self.hidden2tag(bert_out)
        return features  # [batch,seq_len,tag_size]


if __name__ == '__main__':
    from model_utils import custom_local_bert, custom_local_bert_tokenizer
    from config import ArgsParse
    from corpus_proc import read_corpus, generate_dataloader
    from ner_dataset import NerDataset

    # 语料文件
    corpus_dir = os.path.join(os.path.dirname(__file__), 'corpus')
    train_file = os.path.join(corpus_dir, 'trainbio4modelA.txt')
    tags_file = os.path.join(corpus_dir, 'tags.json')

    # # 加载网络模型
    # ckpt = 'bert-base-chinese'
    ckpt = 'chinese-bert-wwm'
    # bert_model, tokenizer = load_transformers_components(ckpt)

    # 加载本地缓存模型目录
    local = os.path.join(os.path.dirname(__file__), 'bert_model/chinese-bert-wwm')
    # 加载定制bert模型
    tokenizer = custom_local_bert_tokenizer(local, max_position=1100)
    bert_model = custom_local_bert(local, max_position=1100)

    opt = ArgsParse().get_parser()
    sentences, sent_tags = read_corpus(opt.train_file)
    dataset = NerDataset(sentences, sent_tags)
    data_loader = generate_dataloader(dataset, tokenizer, opt.tags, 4)

    model = BertCRF(
        bert_model=bert_model,
        target_size=len(opt.tags))
    model.to(device)

    for train_data in data_loader:
        # 模型训练张量注册到device
        input_ids, token_type_ids, attention_mask, label_ids, label_mask = map(lambda x: train_data[x].to(device),
                                                                               train_data)
        # 模型推理
        output = model(input_ids, token_type_ids, attention_mask)
        # 获取模型最后一层的输出
        print(output.shape)
        break
