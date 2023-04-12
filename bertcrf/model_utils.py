#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   bert_model_utils.py
@Time    :   2022/01/24 14:03:49
@Author  :   Mr LaoChen
@Version :   1.0
'''


import os
import torch
from torch.nn.modules.sparse import Embedding 
from bert_crf import BertCRF
from transformers import AutoModel,AutoConfig,AutoTokenizer


class BertHierarchicalPositionEmbedding(Embedding):
    """
    分层位置编码PositionEmbedding
    """
    def __init__(self, alpha=0.4, num_embeddings=512, embedding_dim=768):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.input_dim = num_embeddings
        self.alpha = alpha

    def forward(self, input):
        
        input_shape = input.shape
        seq_len = input_shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.int64).to(input.device)

        embeddings = self.weight - self.alpha * self.weight[:1]
        embeddings = embeddings / (1 - self.alpha)
        embeddings_x = torch.index_select(embeddings, 0, torch.div(position_ids, self.input_dim, rounding_mode='trunc'))
        embeddings_y = torch.index_select(embeddings, 0, position_ids % self.input_dim)
        embeddings = self.alpha * embeddings_x + (1 - self.alpha) * embeddings_y
        return embeddings

def model_local_persist(ckpt, local_dir):
    """
    transformers模型本地化缓存
    """
    _tokenizer = AutoTokenizer.from_pretrained(ckpt)
    _tokenizer.save_pretrained(local_dir)
    # _config = AutoConfig.from_pretrained(ckpt)
    # _config.save_pretrained(local_dir)
    _model = AutoModel.from_pretrained(ckpt)
    _model.save_pretrained(local_dir)
    
    print(ckpt, "transformers模型本地保存成功！")

def generate_position_embedding(bert_model_file):
    """
    通过bert预训练权重创建BertHierarchicalPositionEmbedding并返回
    """
    state_dict = torch.load(bert_model_file)
    # 加载bert预训练文件中的position embedding的weight
    embedding_weight = state_dict['bert.embeddings.position_embeddings.weight']
    hierarchical_position = BertHierarchicalPositionEmbedding()
    hierarchical_position.weight.data.copy_(embedding_weight)
    # 不参与模型训练
    hierarchical_position.weight.requires_grad = False
    return hierarchical_position

def custom_local_bert(local, max_position=512):
    """
    加载本地化缓存transformers模型
    """
    # model file
    model_file = os.path.join(local, 'pytorch_model.bin')
    # config
    config = custom_local_bert_config(local, max_position)
    # load model 忽略模型权重大小不匹配的加载项
    model = AutoModel.from_pretrained(local, config=config, ignore_mismatched_sizes=True)
    # 创建分层position embedding
    hierarchical_embedding = generate_position_embedding(model_file)
    # 新position embedding嵌入现有bert模型
    model.embeddings.position_embeddings = hierarchical_embedding

    return model

def custom_local_bert_config(local, max_position=512):
    """
    加载本地化缓存的transformers模型配置对象
    """
    # model config
    config = AutoConfig.from_pretrained(local, max_position_embeddings=max_position)
    return config

def custom_local_bert_tokenizer(local, max_position=512):
    """
    加载本地化缓存的transformers Tokenizer对象
    """
    # model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local, model_max_length=max_position, )
    return tokenizer

def load_ner_model(opt):
    """
    通过存盘文件加载ner模型
    """
    # 加载模型
    if opt.use == 'A':
        model_file = os.path.join(opt.save_model_dir_A, opt.load_model)
    else:
        model_file = os.path.join(opt.save_model_dir_B, opt.load_model)
    saved_dict = torch.load(model_file)
    ner_model = saved_dict['ner_model']

    # 重建定制postion_id的bert模型
    local = os.path.abspath(os.path.join(opt.local_model_dir, opt.bert_model))
    bert_model = custom_local_bert(local, max_position=opt.max_position_length)
    model = BertCRF(bert_model=bert_model, target_size=len(opt.tags))
    # 加载保存的模型参数
    model.load_state_dict(ner_model)
    return model

def save_ner_model(opt, model, accuracy):
    """
    保存ner模型
    """
    if opt.use == 'A':
        torch.save({
            'ner_model': model.state_dict()
        }, os.path.join(os.path.dirname(__file__), opt.save_model_dir_A, 'ner_model_acc_{:.2f}.pth'.format(accuracy))
        )
    else:
        torch.save({
            'ner_model': model.state_dict()
        }, os.path.join(os.path.dirname(__file__), opt.save_model_dir_B, 'ner_model_acc_{:.2f}.pth'.format(accuracy))
        )

if __name__ == '__main__':
    import os
    # 定制的Bert模型position最大长度参数
    max_position = 1100
    # model checkpoint directory
    local = os.path.join(os.path.dirname(__file__), 'bert_model/bert-base-chinese')
    # 加载定制bert模型
    model = custom_local_bert(local, max_position=max_position)

    print(model)


    # tokenizer = custom_local_bert_tokenizer(local, max_position=max_position)

    # contents = "各五个哈师范的观点" * 200
    # print(f'文本长度:{len(contents)}')

    # inputs = tokenizer([contents],return_tensors='pt')
    # result = model(**inputs)
    # print(f"bert最后一层输出维度:{result['last_hidden_state'].shape}")

    # ckpt = "hfl/chinese-bert-wwm"
    # model_local_persist(ckpt, './bert_model/chinese-bert-wwm')
