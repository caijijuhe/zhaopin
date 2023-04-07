#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ner_trainer.py
@Time    :   2022/01/01 18:47:18
@Author  :   Mr LaoChen
@Version :   1.0
'''

import os
import torch
from tqdm import tqdm
import torchmetrics
from corpus_proc import read_corpus, generate_dataloader
from transformers import AdamW
from ner_dataset import NerDataset
from bert_crf import BertCRF

from model_utils import custom_local_bert, custom_local_bert_tokenizer, load_ner_model, save_ner_model
from config import ArgsParse

import warnings

# 禁用UserWarning
warnings.filterwarnings("ignore")


def get_dataloader(corpus_files, tags, tokenizer, batch_size=16):
    """
    加载语料文件并通过转换为模型用dataloader
    """
    sentences, sent_tags = read_corpus(corpus_files)
    dataset = NerDataset(sentences, sent_tags)
    data_loader = generate_dataloader(dataset, tokenizer, tags, batch_size)
    return data_loader


def train(opt, model, train_dl, test_dl, entity_targets_count):
    """
    模型训练方法
    """
    # 模型优化器
    optimizer = AdamW(model.parameters(), lr=opt.learn_rate)
    # training
    for e in range(opt.epochs):

        # acc = accuracy(opt, model, test_dl, entity_targets_count)

        pbar = tqdm(train_dl)
        model.train()
        for i, batch_data in enumerate(pbar):
            # 模型输入
            batch_data = {k: v.to(opt.device) for k, v in batch_data.items()}
            # logits
            output = model(batch_data['input_ids'], batch_data['token_type_ids'], batch_data['attention_mask'])
            # 计算损失
            loss = model.loss(output, batch_data['label_ids'], batch_data['label_mask'])
            loss = loss / opt.accumulation_steps
            # 计算模型参数梯度
            loss.backward()
            if i % opt.accumulation_steps == opt.accumulation_steps - 1:
                optimizer.step()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
                # 更新梯度
                optimizer.step()
                # 清除累计的梯度值
                model.zero_grad()
                pbar.set_description('Epochs %d/%d loss %f' % (e + 1, opt.epochs, loss.item()))

        acc = accuracy(opt, model, test_dl, entity_targets_count)
        # 每个epoch后保存模型
        save_ner_model(opt, model, acc)


def accuracy(opt, model, test_dl, entity_targets_count):
    with torch.no_grad():  # 不进行反向传播
        metric = torchmetrics.Accuracy(task='multiclass', num_classes=11)
        metric.to(opt.device)
        entity_matches_count = {}
        model.eval()
        pbar = tqdm(test_dl)
        for batch_data in pbar:
            batch_data = {k: v.to(opt.device) for k, v in batch_data.items()}
            outputs = model(batch_data['input_ids'], batch_data['token_type_ids'], batch_data['attention_mask'])
            # 解码
            predicted = model.decode(outputs, batch_data['label_mask'])
            # 解码后结果
            preds = torch.tensor([p for pred in predicted for p in pred], dtype=torch.int64).to(opt.device)
            # 真实结果
            label_ids = batch_data['label_ids']
            target = label_ids[batch_data['label_mask']].flatten()

            assert len(preds) == len(target), '预测标签长度需要和真实标签长度一致'
            # 记录并计算批次中准确率
            metric(preds, target)

            """
            国 务 院 发 布 最 新 通 告
            B  I  I  O  O  O  O  O  O  <真实值>
            B  I  O  O  O  O  O  O  O<预测值> 
            """

            # 真实标签中实体标记匹配
            match_result = entity_matche(preds, target, opt.entity_pair_ix)
            for entity in opt.entity_pair_ix:
                entity_matches_count[entity] = entity_matches_count.get(entity, 0) + match_result.get(entity, 0)

            pbar.set_description('accuracy')

    print(entity_matches_count)
    print(entity_targets_count)
    acc = metric.compute()
    print('Accuracy of the model on test: %.2f %%' % (acc.item() * 100))
    return acc * 100


def entity_matche(preds, target, entity_pair_ix):
    # 真实标签中实体标记匹配
    entity_matchs = {}
    for entity, pair_ix in entity_pair_ix.items():
        i, j = 0, 0
        match_indices = []
        while i < len(target):
            if target[i] == pair_ix[0]:  # 匹配实体标记起始位置
                if i + 1 < len(target):
                    j = i + 1
                    while j < len(target) and target[j] == pair_ix[1]:
                        j += 1
                    match_indices.append((i, j))
                elif i == len(target):
                    match_indices.append((i, i + 1))
            i += 1
        # 预测标签匹配的实体
        for start_idx, end_idx in match_indices:
            for i in range(start_idx, end_idx):
                if preds[i] != target[i]:
                    break
            if i == end_idx - 1:
                entity_matchs[entity] = entity_matchs.get(entity, 0) + 1
    return entity_matchs


def entity_collect(dataloader, entity_pair_ix):
    # 收集标签中不同类别实体数量
    entity_count = {}

    # 统计每个实体数量
    for entity, pair_ix in entity_pair_ix.items():
        # 从测试集寻找当前实体个数
        for batch_data in dataloader:
            label_idx, label_mask = batch_data['label_ids'], batch_data['label_mask']
            target = label_idx[label_mask].flatten()
            i = 0
            while i < len(target):
                # 查找实体起始位置
                if target[i].item() == pair_ix[0]:
                    entity_count[entity] = entity_count.get(entity, 0) + 1
                i += 1
    return entity_count


if __name__ == '__main__':
    # 加载模型相关参数
    opt = ArgsParse.get_parser()
    # 本地模型缓存目录
    local = os.path.join(os.path.dirname(__file__), opt.local_model_dir, opt.bert_model)
    # 加载定制bert模型
    bert_model = custom_local_bert(local, max_position=opt.max_position_length)
    tokenizer = custom_local_bert_tokenizer(local, max_position=opt.max_position_length)

    # 模型语料文件目录
    train_file = os.path.join(os.path.dirname(__file__), opt.train_file)
    dev_file = os.path.join(os.path.dirname(__file__), opt.dev_file)
    test_file = os.path.join(os.path.dirname(__file__), opt.test_file)

    # 模型训练语料
    train_dl = get_dataloader(train_file, opt.tags, tokenizer, batch_size=opt.batch_size)
    test_dl = get_dataloader(test_file, opt.tags, tokenizer)
    entity_count = entity_collect(test_dl, opt.entity_pair_ix)

    # BiLSTMCRF模型
    model = BertCRF(
        bert_model=bert_model,
        target_size=len(opt.tags))

    # # 连续训练，加载之前存盘的模型
    # model = load_ner_model(opt)

    model.to(opt.device)
    # 模型训练
    train(opt, model, train_dl, test_dl, entity_count)
