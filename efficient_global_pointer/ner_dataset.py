#!/usr/bin/env python
# -*- coding:utf-8
'''
@File	:	ner_dataset.py
@Time	:	2022/01/01 10:15:42
@Author	:	Mr LaoChen
@Version:	1.0
'''

import os
import json
from torch.utils.data import Dataset, IterableDataset

class NerDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class NerIterableDataset(IterableDataset):

    def __init__(self, corpus_path):
        self.corpu_file = open(corpus_path,'r', encoding='utf-8')
        self.length = int(self.corpu_file.readline().strip())
        self.start_pos = self.corpu_file.tell()

    def __iter__(self):
        def __iter():
            # 文件读取位置定位到第一行
            self.corpu_file.seek(self.start_pos, 0)
            while True:
                line_str = self.corpu_file.readline().strip()
                # 读取到文件末尾时，终止循环
                if not line_str: break
                yield json.loads(line_str)
        return __iter()

    def __len__(self):
        return self.length

    def __del__(self):
        # 对象销毁前关闭文件释放资源
        if self.corpu_file != None:
            self.corpu_file.close()
            

if __name__ == '__main__':
    from process import generate_dataloader
    corpu_file = os.path.join(os.path.dirname(__file__),'corpus','msra_mid','msra_train.json')
    test_file = os.path.join(os.path.dirname(__file__),'corpus','msra_mid','msra_test.json')

    dataset = NerIterableDataset(corpu_file)
    # dataset = NerDataset(test_file)
    for i in dataset:
        j = i
        # print(i)
        # break