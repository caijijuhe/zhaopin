#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bert_egp.py
@Time    :   2022/02/15 13:25:31
@Author  :   Mr LaoChen
@Version :   1.0
'''

import os
import torch
from torch import nn
from loss_fun import loss_fun
from transformers import BertModel, BertTokenizerFast
from process import load_bio_corpus

from config import ArgsParse


def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)


def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """

    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode  # 指定位置编码向量和输入向量的融合方式（可选的方式有add、mul和zero）
        self.custom_position_ids = custom_position_ids  # 是否使用自定义的位置id，如果为True
        # ，则输入需要是一个包含两个张量的元组，第一个张量是输入张量，第二个张量是自定义的位置id，用于指定每个位置的编码向量。如果为False，则默认使用0到seq_len-1的位置id。

    def forward(self, inputs):

        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            # [batch_size, seq_len, 128]
            input_shape = inputs.shape
            # 提取批次和序列长度
            batch_size, seq_len = input_shape[0], input_shape[1]
            # 创建seq_len长度相同float类型向量  [1,seq_len]
            position_ids = torch.arange(seq_len).type(torch.float).reshape((1, -1))
        """
        第一行代码使用 torch.arange() 创建了一个索引张量。由于这些索引是为 transformer 模型创建的，该模型使用大小为 self.output_dim 
        的连接输入嵌入。结果张量的值将从 0 到 self.output_dim // 2 - 1。

        第二行代码应用了 transformer 模型中使用的位置编码公式。公式为 PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) 对于偶数索引和 
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) 对于奇数索引，其中 pos 是位置，i 是索引。在这种情况下，indices 被用作 i 值，self.output_dim 被用作d_model
        """
        # 编码向量[0.,1.,...32.]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        # transformer中postional编码方式
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        # einsum(爱因斯坦求和公式):通过符号表达对于目标计算或转换操作
        # bn,n 所指代的是后面两个张量的维度 
        # -> 代表转换操作
        # bnd 两个张量计算后结果的维度
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        emb = embeddings
        # 上面计算出三维张量结果，通过sin、cos转换后，stack合并到最后一个维度() [1,seq_len,32,2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        # 变换shap [1,seq_len,64]
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


class EffiGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        # encoder: bert-base
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super(EffiGlobalPointer, self).__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.RoPE = RoPE

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size,
                                 self.ent_type_size * 2)  # 原版的dense2是(inner_dim * 2, ent_type_size * 2)

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        self.device = input_ids.device
        with torch.no_grad():
            context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state
        outputs = self.dense_1(last_hidden_state)
        # 'abcdefg'[::2] = 'ac'
        # 提取所有奇数、偶数位置隐层参数 [batch_size, seq_len, 64]
        qw, kw = outputs[..., ::2], outputs[..., 1::2]  # 从0,1开始间隔为2  [:,:,]  [...,2]

        # 使用旋转位置编码
        if self.RoPE:
            # 位置编码变换矩阵
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            # 提取其中sin、cos编码值扩容到  [1,seq_len,64]
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            # qw拆分为 i和j位置矩阵，通过statck方式穿插组合在一起  [0,2,4,6,..] [1,3,5,7,...]
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1)
            qw2 = torch.reshape(qw2, qw.shape)
            # 位置编码矩阵和qw进行运算，之后求和(位置编码特征系数附加到qw矩阵参数中)
            qw = qw * cos_pos + qw2 * sin_pos
            # kw如法炮制
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # 通过爱因斯坦求和公式，对qw和kw进行dot，之后 sqrt(dim)
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
        # bias偏置计算
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        # bias和 logits合并
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits[:, None] 增加一个维度
        # 屏蔽到下三角推理输出
        logits = self.add_mask_tril(logits, mask=attention_mask)
        # 返回的loss
        out = (logits,)
        # 如果模型还接收了lable参数，计算损失并返回logits和loss
        if labels is not None:
            loss = loss_fun(logits, labels)
            out = (loss,) + out
        return out


if __name__ == '__main__':
    from process import get_dataloader, load_json_corpus
    from ner_dataset import NerDataset

    opt = ArgsParse().get_parser()
    local = os.path.join(os.path.dirname(__file__), opt.local_model_dir, opt.bert_model)
    bert = BertModel.from_pretrained(local)
    tokenizer = BertTokenizerFast.from_pretrained(local)

    # 创建模型对象
    egp = EffiGlobalPointer(bert, opt.categories_size, opt.head_size)
    # print(egp)
    # 模型训练语料
    # 加载语料
    corpus_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), opt.corpus_dir))
    corpus_data = load_bio_corpus(opt)
    # 语料数据集
    dataset = NerDataset(corpus_data)

    dataloader = get_dataloader(opt, dataset, tokenizer)

    for train_data in dataloader:
        input_ids = train_data['input_ids']
        attention_mask = train_data['attention_mask']
        token_type_ids = train_data['token_type_ids']
        labels = train_data['labels']
        print(labels.numpy())
        logits = egp(input_ids, attention_mask, token_type_ids, labels)[0]
        print(logits.shape)
        break
