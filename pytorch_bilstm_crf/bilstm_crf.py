# BiLSTM模型
import torch
from torch import nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, taget_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, 
                            batch_first=True)
        # 将LSTM的输出映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, taget_size)
        # CRF层
        self.crf = CRF(taget_size, batch_first=True)

    def loss(self, out, target, mask):
        return  -1 * self.crf(out, target, mask)

    def decode(self, out, mask):
        return self.crf.decode(out, mask)

    def forward(self, sentence):
        # embedding层输出的张量维度 (batch_size, seq_len,embedding_dim)
        embeds = self.word_embeds(sentence)
        # lstm所有时间步的输出和最后一次输出的hidden_state
        lstm_out, self.hidden = self.lstm(embeds)
        # 调整lstm输出张量维度(seq_len,rnn_hidden_dim)
        # lstm_out = lstm_out[:,-1,:]
        # 线性层运算推理
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

if __name__ == '__main__':
    import os
    from preprocess import build_vocab, build_tag_dict, read_corpus

    EMBEDDING_DIM = 16       
    HIDDEN_DIM = 32

    train_file = os.path.abspath('BIO/msra_train_bio')
    tag_file = os.path.abspath('BIO/tags.txt')

    train_tokens,train_tags = read_corpus(train_file)
    vocab = build_vocab(train_tokens)
    tag_to_ix = build_tag_dict(tag_file)
    bilstm_crf = BiLSTM_CRF(
        vocab_size = len(vocab), 
        embedding_dim = EMBEDDING_DIM,
        hidden_dim = HIDDEN_DIM,
        taget_size = len(tag_to_ix))
    
    print(bilstm_crf)
    