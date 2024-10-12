import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.autograd as autograd
import logging

logger = logging.getLogger(__name__)

import math


class AdditiveAttention(nn.Module):
    """Additive attention.

    Defined in :numref:`subsec_batch_dot`"""
    def __init__(self, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = F.softmax(scores, dim=1)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_batch_dot`"""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = F.softmax(scores, dim=1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, args, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.args = args
        self.num_heads = num_heads
        self.attention = AdditiveAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.fc = nn.Linear(self.args.filter_size * num_hiddens, num_hiddens)

    def forward(self, queries, keys, values):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        batch_size = queries.shape[0]
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # if valid_lens is not None:
        #     # 在轴0，将第一项（标量或者矢量）复制num_heads次，
        #     # 然后如此复制第二项，然后诸如此类。
        #     valid_lens = torch.repeat_interleave(
        #         valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        outputs = self.W_o(output_concat)

        outputs = outputs.reshape(batch_size, -1)
        outputs = self.fc(outputs)
        return outputs

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttentionFuse(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.linear = nn.Linear(self.args.filter_size * config.hidden_size, config.hidden_size)

        self.multi_head_attention = MultiHeadAttention(config.hidden_size, config.hidden_size, config.hidden_size, config.hidden_size, self.args.num_head, config.hidden_dropout_prob, self.args)

    def forward(self, features, **kwargs):
        # ------------- cnn -------------------------
        x = torch.unsqueeze(features, dim=1)  # [B, L*D] -> [B, 1, D*L]
        x = x.reshape(x.shape[0], -1, 768)  # [B, L, D]
        outputs = self.multi_head_attention(x, x, x)  # ->[B, 128]
        # features = self.linear_mlp(features)       # [B, L*D] -> [B, D]
        features = self.linear(features)  # [B, 3*768] -> [B, 768]
        x = torch.cat((outputs, features), dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        # ------------- cnn ----------------------

        # x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        return x


class DistanceClassifier(nn.Module):
    def __init__(self, config, input_size, hidden_size):
        super(DistanceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input1, input2):
        # Concatenate two input embeddings
        x = torch.cat((input1, input2), dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.linear = nn.Linear(3, 1)  # 3->5
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.multi_head_fuse = MultiHeadAttentionFuse(config, self.args)
        self.distance_cal = DistanceClassifier(config, 2 * config.hidden_size, 2 * self.args.d_size)  # [2*768,2*128]

    # 计算代码表示
    def enc(self, seq_embeds):
        batch_size = seq_embeds.shape[0]
        # token_len = seq_ids.shape[-1]
        # # 计算路径表示E                                                    # [batch, path_num, ids_len_of_path/token_len]
        # seq_inputs = seq_ids.reshape(-1, token_len)  # [4, 3, 400] -> [4*3, 400]
        # seq_embeds = self.encoder(seq_inputs, attention_mask=seq_inputs.ne(1))[0]  # [4*3, 400] -> [4*3, 400, 768]
        # seq_embeds = seq_embeds[:, 0, :]  # [4*3, 400, 768] -> [4*3, 768]
        outputs_seq = seq_embeds.reshape(batch_size, -1)  # [4*3, 768] -> [4, 3*768]
        # outputs_seq = self.dropout(outputs_seq)

        # 计算代码表示Z
        return self.multi_head_fuse(outputs_seq)

    def forward(self, anchor, positive, negative=None):
        if negative is not None:
            an_logits = self.enc(anchor)  # [B, 768]
            po_logits = self.enc(positive)
            ne_logits = self.enc(negative)
            ap_dis = self.distance_cal(an_logits, po_logits)
            an_dis = self.distance_cal(an_logits, ne_logits)

            return ap_dis, an_dis
        else:
            an_logits = self.enc(anchor)
            co_logits = self.enc(positive)
            ac_dis = self.distance_cal(an_logits, co_logits)
            return ac_dis
