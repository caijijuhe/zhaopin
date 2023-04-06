import os
import json
from collections import defaultdict

dataset_file = os.path.join(os.path.dirname(__file__), 'dataset')
# 获取所有需要合并的jsonl文件的路径的文件名
input_files = [os.path.join(dataset_file, f)
               for f in os.listdir(dataset_file)
               if f.endswith('.jsonl')]


def generate_json():
    """将标注系统下载下来的文件转换为标准json格式"""
    f1 = open('corpus/corpus.json', 'w', encoding='utf-8')
    f1.write("[")
    file_nums, file_num = len(input_files), 1
    for file_path in input_files:
        with open(file_path, "r", encoding='utf-8') as f2:
            lines = f2.readlines()
            i = 0
            k = len(lines)
            while i < k - 2:
                f1.write(lines[i].strip() + ',\n')
                i += 1
            if file_num < file_nums:
                f1.write(lines[i].strip() + ',\n')
            else:
                f1.write(lines[i].strip() + '\n')
            file_num += 1
    f1.write(']')
    f1.close()


def split_data(data_file, train_ratio=0.8):
    """
    将 JSON 数据对象拆分为训练集和测试集。

    参数:
        data (list): 一个 JSON 对象列表。
        train_ratio (float): 用于训练的数据比例（默认为 0.8）。
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        split_index = int(len(data) * train_ratio)
        train_data = data[:split_index]
        test_data = data[split_index:]
    with open('corpus/train.json', 'w', encoding='utf-8') as f1:
        json.dump(train_data, f1, ensure_ascii=False)
    with open('corpus/test.json', 'w', encoding='utf-8') as f2:
        json.dump(test_data, f2, ensure_ascii=False)


def convert2pointer(input_data_file, output_datafile):
    data = []
    with open(input_data_file, 'r', encoding='utf-8') as f1:
        load = json.load(f1)
        for line in load:
            keywords = defaultdict(list)
            labels = line['label']
            for start, end, label in labels:
                keywords[label].append([start, end])
            data.append({'id': line['id'], 'text': line['text'], 'keywords': keywords})
    with open(output_datafile, 'w', encoding='utf-8') as f2:
        json.dump(data, f2, ensure_ascii=False)


def convert2bio(input_data_file, output_datafile):
    def GenerateLabel(tags: list, label: list):
        tags[label[0]] = 'B-' + str(label[2])
        k = label[0] + 1
        while k < label[1]:
            tags[k] = 'I-' + str(label[2])
            k += 1

    def write2file(tags, file):
        for word, tag in zip(text, tags):
            if word == '\t' or word == ' ' or word == '　':
                continue
            file.write(word + ' ' + tag + '\n')
        file.write("\n")

    convert_fileA = open('corpus/' + str(output_datafile) + '/bio4modelA.txt', 'w', encoding='utf-8')
    convert_fileB = open('corpus/' + str(output_datafile) + '/bio4modelB.txt', 'w', encoding='utf-8')
    with open(input_data_file, 'r', encoding='utf-8') as f:
        load = json.load(f)
        for i in range(len(load)):
            labels = load[i]['label']
            text = load[i]['text'].strip()
            tagsA = ['O'] * len(text)
            tagsB = ['O'] * len(text)
            for j in range(len(labels)):
                label = labels[j]
                if label[2] in ["wor", "sub", "lev", "ben"]:
                    GenerateLabel(tagsA, label)
                else:
                    GenerateLabel(tagsB, label)
            # print(tags)
            write2file(tagsA, convert_fileA)
            write2file(tagsB, convert_fileB)


if __name__ == '__main__':
    # generate_json()
    # split_data('corpus/corpus.json')
    # convert2bio('corpus/train.json', 'trainBIO')
    # convert2bio('corpus/test.json', 'testBIO')
    convert2pointer('corpus/corpus.json', 'corpus/corpus4pointer.json')
