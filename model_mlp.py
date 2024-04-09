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


class MLPFuse(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.linear = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.linear_mlp = nn.Linear(self.args.filter_size * config.hidden_size, config.hidden_size)

    def forward(self, features, **kwargs):
        x_ori = self.linear_mlp(features)
        x = self.dropout(x_ori)
        x = self.dense(x)

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

        self.mlp_fuse = MLPFuse(config, self.args)
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
        return self.mlp_fuse(outputs_seq)

    def forward(self, anchor, positive, negative=None):
        if negative is not None:
            an_logits = self.enc(anchor)        # [B, 768]
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
