# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
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


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x


class TextCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super().__init__()

        self.convs = Conv1d(embedding_dim, n_filters, filter_sizes)   # [768,128,[1,2,3]]
        self.fc = Linear(len(filter_sizes) * n_filters, output_dim)   # [3*128,128]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch size, sent len, emb dim]
        embedded = x.permute(0, 2, 1)  # [B, L, D] -> [B, D, L]
        conved = self.convs(embedded)  # [B, D, L] -> []

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))  # [B, n_filters * len(filter_sizes)]  [B,128*3]
        return self.fc(cat)


class CNNClassificationSeq(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.d_size = self.args.d_size
        self.dense = nn.Linear(2 * self.d_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(config.hidden_size, 1)
        # self.W_w = nn.Parameter(torch.Tensor(2 * config.hidden_size, 2 * config.hidden_size))
        # self.u_w = nn.Parameter(torch.Tensor(2 * config.hidden_size, 1))
        self.linear = nn.Linear(self.args.filter_size * config.hidden_size, self.d_size)
        # self.rnn = nn.LSTM(config.hidden_size, config.hidden_size, 3, bidirectional=True, batch_first=True,
        #                    dropout=config.hidden_dropout_prob)

        # CNN
        self.window_size = self.args.cnn_size
        self.filter_size = []
        for i in range(self.args.filter_size):
            i = i + 1
            self.filter_size.append(i)  # 3
        # self.filter_size.append(3)

        self.cnn = TextCNN(config.hidden_size, self.window_size, self.filter_size, self.d_size, 0.2)

        # self.linear_mlp = nn.Linear(6 * config.hidden_size, self.d_size)
        # self.linear_multi = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, features, **kwargs):
        # ------------- cnn -------------------------
        x = torch.unsqueeze(features, dim=1)  # [B, L*D] -> [B, 1, D*L]
        x = x.reshape(x.shape[0], -1, 768)  # [B, L, D]
        outputs = self.cnn(x)                                 # ->[B, 128]
        # features = self.linear_mlp(features)       # [B, L*D] -> [B, D]
        features = self.linear(features)  # [B, 3*768] -> [B, 128]
        x = torch.cat((outputs, features), dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        # ------------- cnn ----------------------

        # x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.linear = nn.Linear(3, 1)  # 3->5
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.cnnclassifier = CNNClassificationSeq(config, self.args)

    # 计算代码表示
    def enc(self, seq_ids):
        batch_size = seq_ids.shape[0]
        token_len = seq_ids.shape[-1]
        # 计算路径表示E                                                    # [batch, path_num, ids_len_of_path/token_len]
        seq_inputs = seq_ids.reshape(-1, token_len)  # [3, 3, 400] -> [4*3, 400]
        output = self.encoder(seq_inputs, output_hidden_states=True, attention_mask=seq_inputs.ne(1))  # [4*3, 400] -> [4*3, 400, 768]
        seq_embeds = output[0]
        pooler_output = output[1]
        hidden_state = output.hidden_states
        attention = output.attentions
        # seq_embeds = self.encoder(seq_inputs, attention_mask=seq_inputs.ne(1))[0]  # [4*3, 400] -> [4*3, 400, 768]
        # seq_embeds = self.dropout(seq_embeds)
        seq_embeds = seq_embeds[:, 0, :]  # [4*3, 400, 768] -> [4*3, 768]
        outputs_seq = seq_embeds.reshape(batch_size, -1)  # [4*3, 768] -> [4, 3*768]

        # 计算代码表示Z
        # return self.cnnclassifier(outputs_seq)
        return outputs_seq

    def forward(self, anchor, positive, negative=None):
        if negative is not None:
            an_logits = self.enc(anchor)
            po_logits = self.enc(positive)
            ne_logits = self.enc(negative)

            return an_logits, po_logits, ne_logits


# def get_gpu_mem_info(gpu_id=0):
#     """
#     根据显卡 id 获取显存使用信息, 单位 MB
#     :param gpu_id: 显卡 ID
#     :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
#     """
#     import pynvml
#     pynvml.nvmlInit()
#     if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
#         print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
#         return 0, 0, 0
#
#     handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
#     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
#     total = round(meminfo.total / 1024 / 1024, 2)
#     used = round(meminfo.used / 1024 / 1024, 2)
#     free = round(meminfo.free / 1024 / 1024, 2)
#     return total, used, free



# class Model(nn.Module):
#     def __init__(self, encoder, config, tokenizer, args):
#         super(Model, self).__init__()
#         self.encoder = encoder
#         self.config = config
#         self.tokenizer = tokenizer
#         self.args = args
#         self.linear = nn.Linear(3, 1)        # 3->5
#
#         self.cnnclassifier = CNNClassificationSeq(config, self.args)
#
#     def forward(self, seq_ids=None, input_ids=None, labels=None):
#         batch_size = seq_ids.shape[0]
#         seq_len = seq_ids.shape[1]
#         token_len = seq_ids.shape[-1]
#
#         # 计算路径表示E                                                     # [batch, path_num, ids_len_of_path/token_len]
#         seq_inputs = seq_ids.reshape(-1, token_len)                                 # [4, 3, 400] -> [4*3, 400]
#         seq_embeds = self.encoder(seq_inputs, attention_mask=seq_inputs.ne(1))[0]    # [4*3, 400] -> [4*3, 400, 768]
#         seq_embeds = seq_embeds[:, 0, :]                                           # [4*3, 400, 768] -> [4*3, 768]
#         outputs_seq = seq_embeds.reshape(batch_size, -1)                           # [4*3, 768] -> [4, 3*768]
#
#         # 计算代码表示Z
#         logits_path = self.cnnclassifier(outputs_seq)
#
#         # 计算分类结果
#         prob_path = torch.sigmoid(logits_path)
#         prob = prob_path
#         if labels is not None:
#             labels = labels.float()
#             loss = torch.log(prob[:, 0]+1e-10)*labels+torch.log((1-prob)[:, 0]+1e-10)*(1-labels)
#             loss = -loss.mean()
#             return loss, prob
#         else:
#             return prob