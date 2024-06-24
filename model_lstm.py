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

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.d_size = self.args.d_size
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features.reshape(-1, 1536)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class LSTMFuse(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.linear = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.linear_mlp = nn.Linear(self.args.filter_size * config.hidden_size, config.hidden_size)
        self.rnn = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=3, bidirectional=True, batch_first=True,
                           dropout=config.hidden_dropout_prob)

    def forward(self, features, **kwargs):
        x = torch.unsqueeze(features, dim=1)  # [B, L*768] -> [B, 1, L*768]
        x = x.reshape(x.shape[0], -1, 768)
        outputs, hidden = self.rnn(x)  # [10, 3, 2*768] []

        x = outputs[:, -1, :]  # [B, L, 2*D] -> [B, 2*D]
        x = self.linear(x)  # [B, 2*D] -> [B, D]
        x_ori = self.linear_mlp(features)
        x = torch.cat((x, x_ori), dim=-1)

        x = self.dropout(x)
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

        self.lstm_fuse = LSTMFuse(config, self.args)
        self.distance_cal = DistanceClassifier(config, 2 * config.hidden_size, 2 * self.args.d_size)  # [2*768,2*128]
        self.classifier = RobertaClassificationHead(config, args)

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
        return self.lstm_fuse(outputs_seq)

    def forward(self, anchor, positive, labels):
        an_logits = self.enc(anchor)
        co_logits = self.enc(positive)
        input = torch.cat((an_logits, co_logits), dim=1)
        logits=self.classifier(input)
        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
