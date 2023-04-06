#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   trainer.py
@Time    :   2022/02/16 14:40:41
@Author  :   Mr LaoChen
@Version :   1.0
'''
import json
import os

import torch
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from bert_egp import EffiGlobalPointer
from config import ArgsParse
from model_utils import (custom_local_bert, custom_local_bert_tokenizer,
                         load_ner_model, save_ner_model)
from ner_dataset import NerDataset
from process import corpus_split, get_dataloader, load_bio_corpus, entity_collect


def build_optimizer_and_scheduler(opt, model, t_total):
    # 差分学习率
    no_decay = no_decay = ['bias', 'gamma', 'beta']
    model_param = list(model.named_parameters())

    bert_param, other_param = [], []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert':
            bert_param.append((name, para))
        else:
            other_param.append((name, para))

    optimizer_grouped_parameters = [
        # bert模型的差分学习率
        {"params": [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.bert_lr},
        {"params": [p for n, p in bert_param if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.bert_lr},

        # 其他模型层的差分学习率
        {"params": [p for n, p in other_param if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.other_lr},
        {"params": [p for n, p in other_param if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.bert_lr, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def train(opt, model, train_loader, validate_loader):
    t_total = len(train_loader) * opt.epochs
    # 构建模型的optimizer和支持差分学习率的scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(opt, model, t_total)

    # f1_score = evaluate(opt, model, validate_loader)

    for epoch in range(opt.epochs):
        pbar = tqdm(train_loader)
        # 模型训练
        model.train()
        for i, batch_data in enumerate(pbar):
            # 模型训练张量注册到device
            batch_data = {k: v.to(opt.device) for k, v in batch_data.items()}

            loss = model(**batch_data)[0]
            loss = loss / opt.accumulation_steps
            # 计算并更新梯度
            loss.backward()

            # 梯度累加
            # real_batch_size = batch_size * accumlation_steps
            if i % opt.accumulation_steps == opt.accumulation_steps - 1:
                optimizer.step()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
                scheduler.step()
                model.zero_grad()

                pbar.set_description('Epoch: %d/%d loss: %.5f' % (epoch + 1, opt.epochs, loss.item()))
        # 每轮epochs后评价
        f1_score = evaluate(opt, model, validate_loader)
        # 每轮epochs后保存模型
        save_ner_model(opt, model, f1_score)


def global_pointer_f1_score(y_true, y_pred):
    y_pred = torch.greater(y_pred, 0)
    return torch.sum(y_true * y_pred), torch.sum(y_true + y_pred)


def evaluate(opt, model, dataloader):
    size = len(dataloader.dataset)
    model.eval()
    val_loss = 0
    total_n, total_d = 0, 0
    # 预测匹配的实体计数
    entities_count = {k: 0 for k in opt.categories_rev}
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Evaluate'):
            input_ids = data['input_ids'].to(opt.device)
            attention_mask = data['attention_mask'].to(opt.device)
            token_type_ids = data['token_type_ids'].to(opt.device)
            labels = data['labels'].to(opt.device)
            loss, logits = model(input_ids, attention_mask, token_type_ids, labels)
            val_loss += loss.item()

            # 统计正确预测的实体数 [[1],[2],[3],[1]]
            matches = torch.where(labels > 0)
            for c, e, h, v in zip(*matches):
                if logits[c, e, h, v] > 0:
                    entities_count[e.cpu().item()] += 1

            num, den = global_pointer_f1_score(labels, logits)
            total_n += num
            total_d += den
    val_loss /= size
    val_f1 = 2 * total_n / total_d
    print(f"F1:{val_f1:>4f},Avg loss: {val_loss:>5f} \n")
    print("目标实体统计:")
    print(''.join([f'{k}:{v}\n' for k, v in opt.entity_collect.items()]), end='')
    print("预测实体统计:")
    print(''.join([f'{opt.categories_rev[k]}:{v}\n' for k, v in entities_count.items()]), end='')
    return val_f1


def main(opt):
    # 构建模型
    tokenizer = custom_local_bert_tokenizer(opt)
    bert_model = custom_local_bert(opt)

    if opt.load_model != '':
        model, _ = load_ner_model(opt)
        print('模型加载成功!')
    else:
        model = EffiGlobalPointer(bert_model, opt.categories_size, opt.head_size)
        print('模型创建成功!')
    model.to(opt.device)

    # 加载语料
    # corpus_data = load_bio_corpus(opt)
    with open(os.path.join(opt.corpus_dir, opt.corpus_load_file), 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)

    # 拆分语料
    train_corpus, validate_corpus = corpus_split(corpus_data, split_rate=0.95)
    train_loader = get_dataloader(opt, NerDataset(train_corpus), tokenizer)
    validate_loader = get_dataloader(opt, NerDataset(validate_corpus), tokenizer)

    # 验证集实体数量
    opt.entity_collect = entity_collect(opt, validate_loader)

    # 训练模型并保存
    train(opt, model, train_loader, validate_loader)


if __name__ == '__main__':
    opt = ArgsParse().get_parser()
    main(opt)
