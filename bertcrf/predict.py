#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   predict.py
@Time    :   2022/01/23 13:05:01
@Author  :   Mr LaoChen
@Version :   1.0
'''

import os
import torch
from config import ArgsParse
from model_utils import custom_local_bert_tokenizer
from transformers import BertModel
from model_utils import custom_local_bert, load_ner_model


def predict(opt, input_text):
    # 把传入的字符串文本转换为字符串列表
    if input_text[0] == input_text[0][0]:
        input_text = [input_text]

    # 本地模型缓存目录
    local = os.path.abspath(os.path.join(opt.local_model_dir, opt.bert_model))
    # 加载本地缓存的Tokenizer
    tokenizer = custom_local_bert_tokenizer(local, max_position=opt.max_position_length)

    # 汉字符号间添加空格间隔
    sparse_text = [' '.join([c for c in text]) for text in input_text]
    # 生成模型输入数据
    input_data = tokenizer(sparse_text, return_tensors='pt')
    # 加载模型
    model = load_ner_model(opt)
    model.to(opt.device)
    # 模型推理并返回结果
    preds = model_predict(opt, model, input_data)
    # 匹配实体
    matches = []
    for pred, text in zip(preds, input_text):
        entities_matches = []
        for entity, pair in opt.entity_pair_ix.items():
            entities = []
            start, end = 0, 0
            for i, tag in enumerate(pred):
                if tag in pair:
                    if tag == pair[0] and start == 0:
                        start = i
                    if tag == pair[1] and start > 0:
                        end = i
                else:
                    if start > 0:
                        if end > 0:
                            match = text[start - 1:end]
                            entities.append({match: (start - 1, end - 1)})
                        else:
                            match = text[start - 1:start]
                            entities.append({match: (start - 1, start - 1)})
                        start, end = 0, 0
            if len(entities) > 0:
                entities_matches.append({entity: entities})
        matches.append(entities_matches)

    return matches


def model_predict(opt, model, input_data):
    input_data = {k: v.to(opt.device) for k, v in input_data.items()}
    # 模型推理
    model.eval()

    with torch.no_grad():
        output = model(**input_data)
    # 解码
    label_mask = input_data['attention_mask'].bool()
    preds = model.decode(output, label_mask)

    return preds


if __name__ == '__main__':
    opt = ArgsParse().get_parser()

    text = "开展长江生态环境保护民主监督，是以习近平同志为核心的中共中央作出的重大部署。一年来，各民主党派中央、无党派人士积极同国家有关部门及对口省市党委和政府对接沟通，灵活高效开展工作，为党和政府科学决策、有效施策提供了重要参考。"
    result = predict(opt, text)
    print(result)
