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
        conved = self.convs(embedded)  # [B, D, L] -> [B,128,L+1-kernel_size]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))  # [B, n_filters * len(filter_sizes)]  [B,128*3]
        return self.fc(cat)


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


class CNNFuse(nn.Module):
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
        # (768,128,3,128)
        # self.linear_mlp = nn.Linear(6 * config.hidden_size, self.d_size)
        # self.linear_multi = nn.Linear(config.hidden_size, config.hidden_size)

    # def forward(self, features, **kwargs):
    #     # ------------- cnn -------------------------
    #     x = torch.unsqueeze(features, dim=1)  # [B, L*D] -> [B, 1, D*L]
    #     x = x.reshape(x.shape[0], -1, 768)  # [B, L, D]
    #     outputs = self.cnn(x)                                 # ->[B, 128]
    #     # features = self.linear_mlp(features)       # [B, L*D] -> [B, D]
    #     x = self.dropout(outputs)
    #     x = self.dense(x)
    #     # ------------- cnn ----------------------
    #
    #     # x = torch.tanh(x)
    #     # x = self.dropout(x)
    #     # x = self.out_proj(x)
    #     return x

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

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.linear = nn.Linear(3, 1)  # 3->5
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.cnn_fuse = CNNFuse(config, self.args)
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
        return self.cnn_fuse(outputs_seq)

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

