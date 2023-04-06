import os
import json

corpus_file = os.path.abspath('corpus/gp_ad_7.jsonl')
convert2bio_file4modelA = os.path.abspath('corpus/gp_ad_7bioA.txt')
convert2bio_file4modelB = os.path.abspath('corpus/gp_ad_7bioB.txt')


def generate_json():
    """将标注系统下载下来的文件转换为标准json格式"""
    f1 = open('corpus/gp_ad_7.json', 'w', encoding='utf-8')
    f1.write("[")
    with open(corpus_file, 'r', encoding='utf-8') as f2:
        lines = f2.readlines()
        k = len(lines)
        i = 0
        while i < k - 2:
            f1.write(lines[i].strip() + ',\n')
            i += 1
        f1.write(lines[i].strip() + '\n')
    f1.write(']')
    f1.close()


def convert2bio():
    def GenerateLabel(tags: list, label: list):
        tags[label[0]] = 'B-' + str(label[2])
        k = label[0] + 1
        while k < label[1]:
            tags[k] = 'I-' + str(label[2])
            k += 1

    def write2file(tags, file):
        for word, tag in zip(text, tags):
            file.write(word + '\t' + tag + '\n')
        file.write("\n")

    convert_fileA = open(convert2bio_file4modelA, 'w', encoding='utf-8')
    convert_fileB = open(convert2bio_file4modelB, 'w', encoding='utf-8')
    with open('corpus/gp_ad_7.json', 'r', encoding='utf-8') as f:
        load = json.load(f)
        for i in range(len(load)):
            labels = load[i]['label']
            text = load[i]['text'].strip()
            tagsA = ['O'] * len(text)
            tagsB = ['O'] * len(text)
            for j in range(len(labels)):
                label = labels[j]
                if label[2] in ["工作", "专业", "程度", "福利"]:
                    GenerateLabel(tagsA, label)
                else:
                    GenerateLabel(tagsB, label)
            # print(tags)
            write2file(tagsA, convert_fileA)
            write2file(tagsB, convert_fileB)


if __name__ == '__main__':
    generate_json()
    convert2bio()
