import os
import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from ner_dataset import NerIterableDataset
from model_utils import custom_local_bert_tokenizer
from ner_dataset import NerDataset
from process import generate_dataloader, entity_collect
from bert_span import BertSpan
from torchmetrics import Accuracy
from model_utils import custom_local_bert, save_ner_model, load_ner_model
from torchmetrics import MetricCollection
from torch.utils.tensorboard import SummaryWriter
from config import ArgsParse
import warnings

# 禁用UserWarning
warnings.filterwarnings("ignore")
writer = SummaryWriter(log_dir='nerlog')


def build_optimizer_and_scheduler(opt, model, t_total):
    # 差分学习率、动态学习率
    no_decay = ['bias', 'gamma', 'beta']
    model_param = list(model.named_parameters())

    bert_param = []
    other_param = []

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

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.bert_lr / 10, eps=opt.adam_epsilon)

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def train(opt, model, train_loader, test_loader):
    acc_num = 0
    f1_num = 0
    pre_num = 0
    re_num = 0
    t_total = len(train_loader) * opt.epochs
    # 构建模型的optimizer和支持差分学习率的scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(opt, model, t_total)

    # 模型优化器
    # optimizer = BertAdam(model.parameters(), lr=opt.bert_lr)
    # 混合精度
    if opt.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(opt.epochs):
        pbar = tqdm(train_loader)
        # 模型训练
        model.train()
        # 损失(梯度)累加计数器
        counter = 0
        # 累加损失
        total_loss = 0
        for batch_data in pbar:
            # 模型训练张量注册到device
            batch_data = {k: v.to(opt.device) for k, v in batch_data.items()}

            if opt.use_amp:
                # 混合精度训练
                with torch.cuda.amp.autocast():
                    loss = model(**batch_data)[0]
                    scaler.scale(loss).backward()  # FP16做存储和乘法加速
                    scaler.unscale_(optimizer)  # FP32做累加避免舍入误差
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss = model(**batch_data)[0]  # batch_size 8

                avg_loss = loss / opt.step_train_loss  # 模型更新分成2步实现
                total_loss += avg_loss
                counter += 1

                if counter >= opt.step_train_loss:
                    # 计算并更新梯度
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
                    optimizer.step()
                    total_loss = 0
                    counter = 0
                """
                当counter等于opt.step_train_loss时，说明已经在opt.step_train_loss步骤中收集了足够的平均损失。
                在这一点上，通过调用total_loss.backward()计算损失的梯度。
                然后使用优化器optimizer（这是在其他地方定义的）将梯度应用到模型参数上。
                最后，将total_loss和counter重置为0，以进行下一组opt.step_train_loss步骤的训练循环。
                要注意的是，模型更新操作分成两个步骤实现。这是很常见的做法，因为它可以帮助我们减少显存使用，并确保我们在更新之前计算的损失是一组实例的平均损失。 
                """
            scheduler.step()
            model.zero_grad()

            pbar.set_description('Epoch: %d/%d average loss: %.5f' % (epoch + 1, opt.epochs, loss.item()))
        # 每轮epochs后评价
        Accuracy, F1Score, Precision, Recall = evaluate(opt, model, test_loader)
        writer.add_scalar('test_acc', Accuracy, acc_num)
        writer.add_scalar('test_f1', F1Score, f1_num)
        writer.add_scalar('test_pre', Precision, pre_num)
        writer.add_scalar('test_re', Recall, re_num)

        f1_num += 1
        acc_num += 1
        pre_num += 1
        re_num += 1
        # 每轮epochs后保存模型
        save_ner_model(opt, model, Accuracy)


def evaluate(opt, model, test_dl):
    # 混合精度
    if opt.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    if opt.use == 'A':
        metric_Collection = MetricCollection([
            MulticlassPrecision(task='multiclass', num_classes=9),
            MulticlassRecall(task='multiclass', num_classes=9),
            MulticlassAccuracy(task='multiclass', num_classes=9),
            MulticlassF1Score(task='multiclass', num_classes=9)])
    else:
        metric_Collection = MetricCollection([
            MulticlassPrecision(task='multiclass', num_classes=7),
            MulticlassRecall(task='multiclass', num_classes=7),
            MulticlassAccuracy(task='multiclass', num_classes=7),
            MulticlassF1Score(task='multiclass', num_classes=7)])
    metric_Collection.to(opt.device)
    pbar = tqdm(test_dl)
    # 模型评价
    model.eval()
    # 预测匹配的实体计数
    entities_count = {k: 0 for k in opt.tags}
    for batch_data in pbar:
        batch_data = {k: v.to(opt.device) for k, v in batch_data.items()}
        # 提取到张量
        input_ids = batch_data['input_ids']
        token_type_ids = batch_data['token_type_ids']
        attention_mask = batch_data['attention_mask']
        start_ids = batch_data['start_ids']
        end_ids = batch_data['end_ids']
        # 模型推理
        start_logits, end_logits = model(input_ids, token_type_ids, attention_mask)
        mask = attention_mask[:, 1:-1].bool()
        start_pred = torch.argmax(start_logits, dim=-1, keepdim=True)
        end_pred = torch.argmax(end_logits, dim=-1, keepdim=True)
        # 过滤掉padding
        start_pred = start_pred[mask]
        end_pred = end_pred[mask]
        start_ids = start_ids[mask]
        end_ids = end_ids[mask]
        # 计算准确率
        metric_Collection(start_pred, start_ids)
        metric_Collection(end_pred, end_ids)

        # 筛选非零内容
        start_idx = torch.nonzero(torch.squeeze(start_ids))
        start_pred = torch.squeeze(start_pred)[start_idx]
        end_idx = torch.nonzero(torch.squeeze(end_ids))
        end_pred = torch.squeeze(end_pred)[end_idx]
        # start和end完整预测的内容
        filted = (start_pred > 0) & (end_pred > 0)
        real_pred = start_pred[filted]
        # 预测的实体
        tag_idx = [int(((i + 1) / 2).item()) for i in real_pred]
        for tidx in tag_idx:
            entities_count[opt.tags_rev[tidx]] += 1

        pbar.set_description('Accuracy')
    dict = metric_Collection.compute()
    print(entities_count)
    print(opt.entity_collect)
    print('Accuracy of the model on test: %.2f %%\n' % (dict['MulticlassAccuracy'].item() * 100))
    print('F1Score of the model on test: %.2f %%\n' % (dict['MulticlassF1Score'].item() * 100))
    print('Precision of the model on test: %.2f %%\n' % (dict['MulticlassPrecision'].item() * 100))
    print('Recall of the model on test: %.2f %%\n' % (dict['MulticlassRecall'].item() * 100))
    return dict['MulticlassAccuracy'].item(), dict['MulticlassF1Score'].item(), \
           dict['MulticlassPrecision'].item(), dict['MulticlassRecall'].item()


if __name__ == '__main__':

    opt = ArgsParse.get_parser()

    # 加载本地缓存bert模型
    local = os.path.join(os.path.dirname(__file__), opt.local_model_dir, opt.bert_model)
    bert_model = custom_local_bert(local, max_position=opt.max_position_length)
    tokenizer = custom_local_bert_tokenizer(local, max_position=opt.max_position_length)

    # 训练用数据
    train_file = os.path.join(os.path.dirname(__file__), opt.train_file)
    test_file = os.path.join(os.path.dirname(__file__), opt.test_file)
    train_ds = NerIterableDataset(train_file)
    test_ds = NerDataset(test_file)
    train_dl = generate_dataloader(train_ds, tokenizer, opt.tags, opt.batch_size)
    test_dl = generate_dataloader(test_ds, tokenizer, opt.tags, 2)
    # 统计测试语料中实体数量
    opt.entity_collect = entity_collect(opt, test_ds)

    if len(opt.load_model) > 0:
        # 加载模型
        model = load_ner_model(opt)
    else:
        # 创建模型
        model = BertSpan(
            bert_model=bert_model,
            mid_linear_dims=opt.mid_linear_dims,
            num_tags=len(opt.tags) * 2 + 1
        )

    model.to(opt.device)

    # 训练模型
    train(opt, model, train_dl, test_dl)
