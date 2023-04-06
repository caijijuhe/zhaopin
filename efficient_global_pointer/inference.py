import os
import json
import torch
import numpy as np
from tqdm import tqdm
from config import ArgsParse
from model_utils import load_ner_model, custom_local_bert_tokenizer

def inference(opt, text, model, tokenizer):
    # 获取文本的offset_mapping
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    new_span, entities= [], []
    # 提取token索引
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])
    # 编码后文本token
    model_input = tokenizer(text, return_tensors='pt')
    input_ids = model_input["input_ids"].to(opt.device)
    token_type_ids = model_input["token_type_ids"].to(opt.device)
    attention_mask = model_input["attention_mask"].to(opt.device)
    # 模型推理
    with torch.no_grad():
        scores = model(input_ids, attention_mask, token_type_ids)[0]
        scores = scores.squeeze(0).cpu().numpy()
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entities.append({"start_idx":new_span[start][0], "end_idx":new_span[end][-1], "type":opt.categories_rev[l]})

    return {"text":text, "entities":entities}

if __name__ == '__main__':
    from process import load_bio_corpus, corpus_split, get_dataloader, NerDataset
    opt = ArgsParse().get_parser()

    # 加载模型和tokenizer
    model,max_length = load_ner_model(opt)
    model.to(opt.device)
    tokenizer = custom_local_bert_tokenizer(opt,max_length)

    # 从语料中随机抽取样本进行推理
    all_ = []
     # 加载语料
    corpus_data = load_bio_corpus(opt)
    
    # 拆分语料
    _, test_corpus = corpus_split(corpus_data, split_rate=0.998)
    for d in tqdm(test_corpus):
        all_.append(inference(opt,d["title"], model, tokenizer))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__),'./outputs/ner_test.json'))
    json.dump(
        all_,
        open(output_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )