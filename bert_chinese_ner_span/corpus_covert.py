#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   corpus_covert.py
@Time    :   2022/01/16 09:01:23
@Author  :   Mr LaoChen
@Version :   1.0
'''

import os
import codecs
import json
import numpy as np
import logging as log

log.basicConfig(level=log.NOTSET)

# 语料文件目录
corpus_path = os.path.join(os.path.dirname(__file__), 'corpus', 'msra')
# 训练集文件路径
train_file = os.path.join(corpus_path, 'trainbio4modelA.txt')
# 测试集文件路径
test_file = os.path.join(corpus_path, 'testbio4modelA.txt')
# 转换前tags标签文件
tags_file = os.path.join(corpus_path, 'tag.txt')

# 转换后语料存盘目录
converted_path = os.path.join(os.path.dirname(__file__), 'corpus', 'msra_mid')


def read_tagclass(tag_file):
    """读取标签"""
    tag_class = set()
    with codecs.open(tags_file, 'r', encoding='utf-8') as f:
        for tag in f.read().split('\n'):
            if tag != 'O':
                tag_class.add(tag.split('-')[1])
    return tag_class


def save_convert_tags(tag_class, converted_tag_file):
    """转换后的标签存盘"""
    with codecs.open(converted_tag_file, 'w', encoding='utf-8') as f:
        tags = {t: i + 1 for i, t in enumerate(tag_class)}
        json.dump(tags, f)

        log.info('tag文件保存成功!')


def convert_corpus(tag_class, corpus_file, converted_file, write_lines=True):
    """传统标注转换为范围标注"""
    sents = []
    tags = []
    with codecs.open(corpus_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sent, tag = [], []
        for line in lines:
            # 语句使用空行分隔
            if len(line.strip()) > 0 and line.strip() != '0':
                w, t = line.split()
                tag.append(t)
                sent.append(w)
            if line.strip() == '':
                # if len(sent) < 510:
                sents.append(sent)
                tags.append(tag)
                sent, tag = [], []

    # 转换语料资料存入json
    json_data = []
    for id, sent in enumerate(sents):
        text = ''.join(sent)
        tag = np.array(tags[id])
        # 所有非‘O’字符为True
        filters = (tag != 'O')

        start = -1
        cls = ''
        tag_cls = {t: [] for t in tag_class}  # 数据格式｛'PER':[[0,2],[11,15]],'LOC':[],'ORG':[]｝
        for i, f in enumerate(filters):
            # 记录起始位置
            if f and start < 0:
                cls = tags[id][i].split('-')[1]
                start = i
            # 记录结束位置 
            if (f == False) and start >= 0 and cls != '':
                end = i - 1
                tag_cls[cls].append([start, end])
                start = -1
                cls = ''
        json_data.append({"id": id, "text": text, "labels": tag_cls})

    with codecs.open(converted_file, 'w', encoding='utf-8') as file:
        if write_lines:
            # 文件首行写入总记录数
            file.write(str(len(json_data)) + '\n')
        for data in json_data:
            json_string = json.dumps(data)
            file.write(json_string + '\n')
    log.info('%s转换成功！' % corpus_file)


if __name__ == '__main__':
    converted_tags_file = os.path.join(converted_path, 'tags.json')
    converted_train_file = os.path.join(converted_path, 'msra_train.json')
    converted_test_file = os.path.join(converted_path, 'msra_test.json')

    tag_class = read_tagclass(tags_file)
    convert_corpus(tag_class, train_file, converted_train_file)
    convert_corpus(tag_class, test_file, converted_test_file, write_lines=False)
