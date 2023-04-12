# 模型训练
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import torchmetrics
from torchmetrics import F1Score,Accuracy,MetricCollection, Recall, Precision
from torchmetrics.classification import MulticlassAccuracy,MulticlassF1Score,MulticlassPrecision,MulticlassRecall
from bilstm_crf import BiLSTM_CRF
from ner_dataset import NerDataSet
from preprocess import build_vocab, build_tag_dict, read_corpus, build_data_loader
import logging as log
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

from config import ArgsParse

writer = SummaryWriter(log_dir='nerlog')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log.basicConfig(level=log.INFO)


def build_optimizer_and_scheduler(opt, model, t_total):
    # 动态学习率
    optimizer = AdamW(model.parameters(), lr=opt.lr / 10, eps=opt.adam_epsilon, weight_decay=opt.weight_decay)

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def train_model(model, train_dl, test_dl, entity_count, entity_pair_ix):
    t_total = len(train_dl) * opt.epochs
    # 模型训练优化器
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer, scheduler = build_optimizer_and_scheduler(opt, model, t_total)

    running_loss = 0.0
    loss_num = 0
    acc_num = 0
    weight_num = 0
    f1_num = 0
    pre_num = 0
    re_num = 0
    # 训练
    for epoch in range(opt.epochs):
        for i, (token_idx, tag_idx, mask) in enumerate(train_dl):
            # Step 1. 清除累计的梯度值
            model.zero_grad()

            # Step 3. 运行前向运算
            out = model(token_idx)
            loss = model.loss(out, tag_idx, mask)

            # Step 4. 计算损失，梯度后通过optimizer更新模型参数
            loss.backward()
            # if i % opt.accumulation_steps == opt.accumulation_steps - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            if i % 10 == 9 or i == len(train_dl):
                print('epoch: %d\t batch: %d\t loss: %.6f' % (epoch + 1, i + 1, running_loss / 100))
                writer.add_scalar('train_loss', running_loss / 100, loss_num)
                writer.add_histogram('linear_hist', model.hidden2tag.get_parameter('weight').flatten(), weight_num)
                running_loss = 0
                weight_num += 1
                loss_num += 1
            if i % 20 == 19 or i == len(train_dl):
                Accuracy, F1Score, Precision, Recall = accuracy(opt, model, test_dl, entity_count, entity_pair_ix)
                writer.add_scalar('test_acc', Accuracy, acc_num)
                writer.add_scalar('test_f1', F1Score, f1_num)
                writer.add_scalar('test_pre', Precision, pre_num)
                writer.add_scalar('test_re', Recall, re_num)

                f1_num += 1
                acc_num += 1
                pre_num += 1
                re_num += 1

    if opt.use == 'A':
            torch.save(model.state_dict(),
                       os.path.join(os.path.dirname(__file__), opt.load_model_A, 'modelA_acc_{:.2f}.pth'.format(Accuracy)))
    else:
            torch.save(model.state_dict(),
                       os.path.join(os.path.dirname(__file__), opt.load_model_B, 'modelB_acc_{:.2f}.pth'.format(Accuracy)))


def accuracy(opt, model, test_dl, entity_targets_count, entity_pair_ix):
    model.eval()
    with torch.no_grad():  # 不进行反向传播
        if opt.use == 'A':
            metric_Collection = torchmetrics.MetricCollection([
                MulticlassPrecision(num_classes=11, average='micro'),
                MulticlassRecall(num_classes=11, average='micro'),
                MulticlassAccuracy(num_classes=11, average='micro'),
                MulticlassF1Score(num_classes=11, average='micro')])
        else:
            metric_Collection = torchmetrics.MetricCollection([
                MulticlassPrecision(num_classes=9, average='micro'),
                MulticlassRecall(num_classes=9, average='micro'),
                MulticlassAccuracy(num_classes=9, average='micro'),
                MulticlassF1Score(num_classes=9, average='micro')])
        metric_Collection.to(device)
        entity_matches_count = {}
        for token_idx, tag_idx, mask in test_dl:
            outputs = model(token_idx)
            predicted = model.decode(outputs, mask)

            preds = torch.tensor([p for pred in predicted for p in pred], dtype=torch.int64).to(device)
            target = tag_idx[mask].flatten()

            assert len(preds) == len(target), '预测标签长度需要和真实标签长度一致'
            # 记录并计算批次中准确率
            metric_Collection(preds, target)

            # 真实标签中实体标记匹配
            match_result = entity_matche(preds, target, entity_pair_ix)
            for entity in entity_pair_ix:
                entity_matches_count[entity] = entity_matches_count.get(entity, 0) + match_result.get(entity, 0)
    model.train()
    print(entity_matches_count)
    print(entity_targets_count)
    dict = metric_Collection.compute()
    print('Accuracy of the model on test: %.2f %%\n' % (dict['MulticlassAccuracy'].item() * 100))
    print('F1Score of the model on test: %.2f %%\n' % (dict['MulticlassF1Score'].item() * 100))
    print('Precision of the model on test: %.2f %%\n' % (dict['MulticlassPrecision'].item() * 100))
    print('Recall of the model on test: %.2f %%\n' % (dict['MulticlassRecall'].item() * 100))
    return dict['MulticlassAccuracy'].item(), dict['MulticlassF1Score'].item(), \
           dict['MulticlassPrecision'].item(), dict['MulticlassRecall'].item()


def entity_collect(dataloader, entity_pair_ix):
    # 收集标签中不同类别实体数量
    entity_count = {}

    for entity, pair_ix in entity_pair_ix.items():
        for _, tag_idx, mask in dataloader:
            target = tag_idx[mask].flatten()
            i = 0
            while i < len(target):
                # 查找实体起始位置
                if target[i].item() == pair_ix[0]:
                    entity_count[entity] = entity_count.get(entity, 0) + 1
                i += 1
    return entity_count


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
                    # while target[j] == pair_ix[1] and j < len(target):
                    while j < len(target) and target[j] == pair_ix[1]:
                        j += 1
                    match_indices.append((i, j))
                elif i + 1 == len(target):
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


if __name__ == '__main__':
    opt = ArgsParse().get_parser()

    # 模型参数
    EMBEDDING_DIM = 16
    HIDDEN_DIM = 32
    # 语料文件_A
    if opt.use == 'A':
        train_file = os.path.abspath(opt.corpus_dir_train_A)
        test_file = os.path.abspath(opt.corpus_dir_test_A)
        tag_file = os.path.abspath(opt.corpus_dir_tag_A)
    else:
        train_file = os.path.abspath(opt.corpus_dir_train_B)
        test_file = os.path.abspath(opt.corpus_dir_test_B)
        tag_file = os.path.abspath(opt.corpus_dir_tag_B)

    # 创建训练用语料
    train_tokens, train_tags = read_corpus(train_file)
    test_tokens, test_tags = read_corpus(test_file)

    # 构建词汇表和tag字典
    vocab = build_vocab(train_tokens)
    tag_to_ix = build_tag_dict(tag_file)
    # 自定义Dataset
    train_ds = NerDataSet(train_tokens, train_tags, vocab, tag_to_ix)
    test_ds = NerDataSet(test_tokens, test_tags, vocab, tag_to_ix)
    # 构建DataLoader
    train_dl = build_data_loader(train_ds, batch_size=opt.batch_size, shuffle=True)
    test_dl = build_data_loader(test_ds, batch_size=4)
    # 创建模型
    model = BiLSTM_CRF(
        vocab_size=len(vocab),
        embedding_dim=opt.embedding_dim,
        hidden_dim=opt.hidden_dim,
        taget_size=len(tag_to_ix))
    if len(opt.load_modelfile) > 0:
        if opt.use == 'A':
            state_dict = torch.load(os.path.join(opt.load_model_A, opt.load_modelfile))
            model.load_state_dict(state_dict)
        else:
            state_dict = torch.load(os.path.join(opt.load_model_B, opt.load_modelfile))
            model.load_state_dict(state_dict)

    model.to(device)

    if opt.use == 'A':
        entity_pair_ix = {
            'job': [tag_to_ix['B-job'], tag_to_ix['I-job']],  # 'ORG':[1,5]
            'major': [tag_to_ix['B-major'], tag_to_ix['I-major']],
            'extent': [tag_to_ix['B-extent'], tag_to_ix['I-extent']],
            'wealthfare': [tag_to_ix['B-wealthfare'], tag_to_ix['I-wealthfare']]
        }
    else:
        entity_pair_ix = {
            'knowledge': [tag_to_ix['B-knowledge'], tag_to_ix['I-knowledge']],
            'skill': [tag_to_ix['B-skill'], tag_to_ix['I-skill']],
            'quality': [tag_to_ix['B-quality'], tag_to_ix['I-quality']]
        }

    # 加载或统计类别数量字典
    if opt.use == 'A':
        if os.path.exists(opt.entity_collect_file_A):
            log.info('entity_count加载成功!')
            entity_count = torch.load(opt.entity_collect_file_A)
        else:
            # 统计数据集中各类别实体数量
            entity_count = entity_collect(test_dl, entity_pair_ix)
            torch.save(entity_count, opt.entity_collect_file_A)
    else:
        if os.path.exists(opt.entity_collect_file_B):
            log.info('entity_count加载成功!')
            entity_count = torch.load(opt.entity_collect_file_B)
        else:
            # 统计数据集中各类别实体数量
            entity_count = entity_collect(test_dl, entity_pair_ix)
            torch.save(entity_count, opt.entity_collect_file_B)

    train_model(model, train_dl, test_dl, entity_count, entity_pair_ix)
