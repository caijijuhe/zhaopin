

import os
import torch
from transformers import BertModel,BertConfig,BertTokenizerFast
from torch.nn.modules.sparse import Embedding 
from bert_egp import EffiGlobalPointer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertHierarchicalPositionEmbedding(Embedding):
    """
    分层位置编码PositionEmbedding
    """
    def __init__(self, alpha=0.4, num_embeddings=512,embedding_dim=768):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.input_dim = num_embeddings
        self.alpha = alpha

    def forward(self, input):
        input_shape = input.shape
        seq_len = input_shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.int64).to(device)

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
    _tokenizer = BertTokenizerFast.from_pretrained(ckpt)
    _tokenizer.save_pretrained(local_dir)
    _model = BertModel.from_pretrained(ckpt)
    _model.save_pretrained(local_dir)
    _config = BertConfig.from_pretrained(ckpt)
    _config.save_pretrained(local_dir)
    print(ckpt, "本地保存成功！")

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

def custom_local_bert(opt, max_length=None):
    """
    加载本地化缓存bert模型并定制最大输入长度
    """
    local = os.path.join(os.path.dirname(__file__),opt.local_model_dir, opt.bert_model)
    max_length = opt.max_length if max_length is None else max_length
    # model file
    model_file = os.path.join(local, 'pytorch_model.bin')
    # model config
    config =custom_local_bert_config(opt, max_length)
    # load model 忽略模型权重大小不匹配的加载项
    model = BertModel.from_pretrained(local, config=config, ignore_mismatched_sizes=True)
    if max_length > 512:
        # 创建分层position embedding
        hierarchical_embedding = generate_position_embedding(model_file)
        # 新position embedding嵌入现有bert模型
        model.embeddings.position_embeddings = hierarchical_embedding
    return model 

def custom_local_bert_config(opt, max_length=None):
    """
    加载本地化缓存的transformers模型配置对象
    """
    local = os.path.join(os.path.dirname(__file__),opt.local_model_dir, opt.bert_model)
    max_length = opt.max_length if max_length is None else max_length
    # model config
    config = BertConfig.from_pretrained(local, max_position_embeddings=max_length)
    return config

def custom_local_bert_tokenizer(opt, max_length=None):
    """
    加载本地化缓存的transformers Tokenizer对象
    """
    local = os.path.join(os.path.dirname(__file__),opt.local_model_dir, opt.bert_model)
    max_length = opt.max_length if max_length is None else max_length
    # model tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(local, model_max_length=max_length)
    return tokenizer

def load_ner_model(opt):
    """
    通过存盘文件加载ner模型
    """
    # 加载模型
    model_file = os.path.join(os.path.dirname(__file__), opt.save_model_dir, opt.load_model)
    saved_dict = torch.load(model_file)
    ent_type_size = saved_dict['ent_type_size']
    # encoder = saved_dict['bert_model']
    inner_dim = saved_dict['inner_dim']
    dense_1 = saved_dict['dense_1']
    dense_2 = saved_dict['dense_2']
    max_length = saved_dict['max_length']

    # 重建定制postion_id的bert模型
    bert_model = custom_local_bert(opt,max_length)
    model = EffiGlobalPointer(
        encoder=bert_model, 
        ent_type_size=ent_type_size, 
        inner_dim=inner_dim)
    # 加载保存的模型参数
    # model.encoder.load_state_dict(encoder)
    model.dense_1.load_state_dict(dense_1)
    model.dense_2.load_state_dict(dense_2)
    return model,max_length

def save_ner_model(opt, model, f1_score):
    """
    保存egp模型
    """
    torch.save({
        'ent_type_size':model.ent_type_size,
        # 'bert_model':model.encoder.state_dict(),
        'inner_dim':model.inner_dim,
        'dense_1':model.dense_1.state_dict(),
        'dense_2':model.dense_2.state_dict(),
        'max_length': opt.max_length
    }, os.path.join(os.path.dirname(__file__), opt.save_model_dir, 'ner_model_f1_{:.2f}.pth'.format(f1_score))
    )    

if __name__ == '__main__':
    import os
    from config import ArgsParse
    opt = ArgsParse().get_parser()

    # 定制的Bert模型position最大长度参数
    max_position = 512
    # model checkpoint directory
    local = os.path.join(os.path.dirname(__file__), 'bert_model/bert-base-chinese')
    # 加载定制bert模型
    model = custom_local_bert(opt)
    tokenizer = custom_local_bert_tokenizer(opt)
    model.to(device)

    contents = "阿斯顿法国红酒" # * 200
    print(f'文本长度:{len(contents)}')

    inputs = tokenizer([contents],return_tensors='pt')
    inputs = {k:v.to(device) for k,v in inputs.items()}
    result = model(**inputs)
    print(f"bert最后一层输出维度:{result['last_hidden_state'].shape}")
