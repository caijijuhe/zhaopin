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

    def __init__(self, corpus_path):
        self.corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline().strip()
                if not line: break
                self.corpus.append(json.loads(line))

    def __getitem__(self, index):
        return self.corpus[index]

    def __len__(self):
        return len(self.corpus)


class NerIterableDataset(IterableDataset):

    def __init__(self, corpus_path):
        self.corpu_file = open(corpus_path, 'r', encoding='utf-8')
        self.length = int(self.corpu_file.readline().strip())
        self.start_pos = self.corpu_file.tell()
        """
        实现对文件指针的移动，文件对象提供了 tell() 函数和 seek() 函数。tell() 函数用于判断文件指针当前所处的位置，而 seek() 函数用于移动文件指针到文件的指定位置。 
        """
    def __iter__(self):
        def __iter():
            # 文件读取位置定位到第一行
            """
            seek() 函数用于将文件指针移动至指定位置，该函数的语法格式如下：

            file.seek(offset[, whence])
            其中，各个参数的含义如下：

            file：表示文件对象；
            whence：作为可选参数，用于指定文件指针要放置的位置，该参数的参数值有 3 个选择：0 代表文件头（默认值）、1 代表当前位置、2 代表文件尾。
            """
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

    corpu_file = os.path.join(os.path.dirname(__file__), 'corpus', 'msra_mid', 'msra_train.json')
    test_file = os.path.join(os.path.dirname(__file__), 'corpus', 'msra_mid', 'msra_test.json')

    dataset = NerIterableDataset(corpu_file)
    # dataset = NerDataset(test_file)
    for i in dataset:
        j = i
        # print(i)
        # break
