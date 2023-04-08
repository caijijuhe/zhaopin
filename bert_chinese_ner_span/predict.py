#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   predict.py
@Time    :   2022/01/27 22:24:53
@Author  :   Mr LaoChen
@Version :   1.0
'''

import os
import torch
from config import ArgsParse
from model_utils import custom_local_bert_tokenizer
from model_utils import load_ner_model

def predict(opt, text):
    """
    识别文本中的实体信息并返回
    """
    if text[0] == text[0][0]:
        text = [text]
    # 文本以空格分隔
    input_text = [' '.join([c for c in t]) for t in text]
    # 加载tokenizer
    local = os.path.join(os.path.dirname(__file__), opt.local_model_dir, opt.bert_model)
    tokenizer = custom_local_bert_tokenizer(local, max_position=opt.max_position_length)
    # 加载模型
    model = load_ner_model(opt)
    model.to(opt.device)
    # 模型推理
    input_data = tokenizer(input_text, padding=True, return_tensors='pt')
    input_data = { k:v.to(opt.device) for k,v in input_data.items()}
    start_pred,end_pred = model_predict(opt, model, input_data)
    start_pred = start_pred.squeeze()
    end_pred = end_pred.squeeze()

    matches = []
    # 遍历预测结果和语料集合
    for start,end,txt in zip(start_pred, end_pred, text):
        match_entities = {tag:[] for tag in opt.tags}
        for i in range(len(start)):
            s = start[i] 
            if s > 0:
                tag_idx = int((s + 1) / 2)
                for j in range(i,len(end)):
                    if end[j] > 0 and (end[j] / 2) == tag_idx:
                        # 实体类别
                        tag_cls = opt.tags_rev[tag_idx]
                        entity = txt[i:j+1]
                        match_entities[tag_cls].append({entity:(i,j)})
                        break
        matches.append(match_entities)
    return matches

def model_predict(opt, model, input_data):
    # 模型推理
    model.eval()
    with torch.no_grad():
        start_logits,end_logits  = model(**input_data)
        start_pred = torch.argmax(start_logits, dim=-1, keepdim=True)
        end_pred = torch.argmax(end_logits, dim=-1, keepdim=True)
        
        return start_pred.cpu().numpy(), end_pred.cpu().numpy() 

def main():
    opt = ArgsParse().get_parser()
    input_text = [
        '本报北京6月14日讯新华社记者尹祝鸿、本报记者毕忠全报道',
        '人民日报肩负宣传邓小平建设有中国特色社会主义理论和党的基本路线']
    result = predict(opt, input_text)
    print(result)

if __name__ == '__main__':
    main()
    



