# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
sys.path.append('/home/EPVD/')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import json
from sklearn.metrics import recall_score,precision_score,f1_score
from tqdm import tqdm, trange
import multiprocessing
from model import Model
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

sys.path.append('..')

import parserTool.parse as ps
from c_cfg import C_CFG
from parserTool.utils import remove_comments_and_docstrings
from parserTool.parse import Lang
import pickle


# 将路径序列转换为代码序列
def extract_pathtoken(source, path_sequence):
    seqtoken_out = []
    for path in path_sequence:
        seq_code = ''
        for line in path:
            if line != 'exit' and (line in source):
                seq_code += source[line]
        seqtoken_out.append(seq_code)
        if len(seqtoken_out) > 5:
            break
    if len(path_sequence) == 0:
        seq_code = ''
        for i in source:
            seq_code += source[i]
        seqtoken_out.append(seq_code)
    seqtoken_out = sorted(seqtoken_out, key=lambda i: len(i), reverse=False)
    return seqtoken_out


class CloneFeatures(object):
    def __init__(self,
                 an_path_embeds,
                 po_path_embeds,
                 ne_path_embeds,
                 ):
        self.an_path_embeds = an_path_embeds
        self.po_path_embeds = po_path_embeds
        self.ne_path_embeds = ne_path_embeds


def convert_examples_to_features_clone(js, tokenizer, path_dict, args):
    codes_paths = []  # 3*3*……
    for i in [1, 2, 3]:
        # clean_code, code_dict = remove_comments_and_docstrings(js[f'code{i}'], 'java')

        # source
        # pre_code = ' '.join(clean_code.split())
        # code_tokens = tokenizer.tokenize(pre_code)[:args.block_size - 2]
        # source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        # source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        # padding_length = args.block_size - len(source_ids)
        # source_ids += [tokenizer.pad_token_id] * padding_length

        # paths
        # g = C_CFG()
        # code_ast = ps.tree_sitter_ast(clean_code, Lang.JAVA)
        # s_ast = g.parse_ast_file(code_ast.root_node)
        # num_path, cfg_allpath, _, _ = g.get_allpath()
        # path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
        path_embeds, cfg_allpath = path_dict[js[f'idx{i}']]
        path_embeds = torch.tensor(path_embeds, dtype=torch.float32)
        
        codes_paths.append(path_embeds)
    return CloneFeatures(codes_paths[0], codes_paths[1], codes_paths[2])


# 输入为未预处理excel生成的jsonl。
# class JsonDataset(Dataset):
class ExcelDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        # pkl_file = open(args.pkl_file, 'rb')
        # path_dict = pickle.load(pkl_file)
        pkl_file = open(args.pkl_file, 'rb')
        path_dict = pickle.load(pkl_file)

        with open(file_path, encoding="UTF-8") as f1:
            for idx, line in enumerate(f1):
                js = json.loads(line.strip())
                # idx_list = line.strip().split(',')
                # for idx in idx_list:
                #     code = open(f"../test_dataset/id2sourcecode/{idx}.java", encoding='UTF-8').read()
                #     group.append(code)
                self.examples.append(convert_examples_to_features_clone(js, tokenizer, path_dict, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (self.examples[i].an_path_embeds,
                self.examples[i].po_path_embeds,
                self.examples[i].ne_path_embeds)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer):
    """ Train the model """ 
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    args.max_steps = args.epoch*len(train_dataloader)
    t_total = args.max_steps // args.gradient_accumulation_steps
    args.save_steps = len(train_dataloader) # 有多少个batch
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total*0.1,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    best_f1 = 0.0
    model.zero_grad()
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            # inputs = batch[0].to(args.device)      # 源码生成的ids
            # labels = batch[1].to(args.device)      # label
            # seq_inputs = batch[2].to(args.device)  # 各路径生成的ids
            anchor = batch[0].to(args.device)
            positive = batch[1].to(args.device)
            negative = batch[2].to(args.device)
            model.train()
            ap_dis, an_dis = model(anchor, positive, negative)  # [Batchsize,768]

            margin = 1
            losses = F.relu(ap_dis - an_dis + margin)
            loss = losses.mean()

            # 计算triplet loss
            # cos_pos = 0.5 - (nn.CosineSimilarity(dim=1)(an_logits, po_logits) * 0.5)  # [Batchsize]
            # cos_neg = 0.5 - (nn.CosineSimilarity(dim=1)(an_logits, ne_logits) * 0.5)
            # margin = 1
            # losses = F.relu(cos_pos - cos_neg + margin)   # [2]
            # loss = losses.mean()   # tensor(1.0089)

            # logger.info("**********")
            # logger.info(f"{type(cos_pos)}")
            # logger.info(f"{type(losses)}")
            # logger.info(f"{type(loss)}")
            # 2*768维向量

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            loss = loss * args.gradient_accumulation_steps
            # DONE 这里的loss需要求平均吗？ 不用
            # DONE 可以求一下当前epochs的平均，方便看出整体变化趋势
            tr_num += 1
            train_loss += loss.item()
            # if avg_loss == 0:
            #     avg_loss = tr_loss
            # avg_loss = round(train_loss/tr_num, 5)
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            global_step += 1

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == 20000:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                # output_flag = True
                # avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     logging_loss = tr_loss
                #     tr_nb = global_step

                # 一个epoch学完进入以下条件
                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            results = evaluate(args, model, tokenizer, idx, eval_when_training=True)
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value, 4))
            # Save model checkpoint
        if results['F1'] > best_f1:
        #if results['eval_acc'] > best_acc:
            best_f1 = results['F1']
            best_precision = results['precision']
            best_recall = results['recall']
            best_threshold = results['threshold']
            logger.info("  "+"*"*20)
            logger.info("  Best f1:%s", round(best_f1, 4))
            logger.info("  "+"*"*20)
            logger.info("  Recall:%s", best_recall)
            logger.info("  " + "*" * 20)
            logger.info("  Precision:%s", best_precision)
            logger.info("  " + "*" * 20)
            logger.info("  threshold:%s", best_threshold)
            logger.info("  " + "*" * 20)


            checkpoint_prefix = 'checkpoint-best-acc'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


import matplotlib.pyplot as plt


def evaluate(args, model, tokenizer, idx, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = ExcelDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    cos_right = []
    cos_wrong = []
    for batch in eval_dataloader:
        anchor = batch[0].to(args.device)
        positive = batch[1].to(args.device)
        negative = batch[2].to(args.device)
        with torch.no_grad():
            ap_dis, an_dis = model(anchor, positive, negative)
        cos_right += ap_dis.tolist()
        cos_wrong += an_dis.tolist()
    temp_best_f1 = 0
    temp_best_recall = 0
    temp_best_precision = 0
    temp_count = 0
    temp_error_count = 0
    temp_error_total = 0
    temp_total = 0
    temp_best_threshold = 0

    count = 0
    error_count = 0
    threshold = 0.5
    for h in cos_right:
        if h[0] <= threshold:
            count += 1  # 实测克隆对个数TP
    total = len(cos_right)  # 所有潜在克隆对（应是克隆对）个数TP+FN
    for h in cos_wrong:
        if h[0] > threshold:
            error_count += 1  # 实测非克隆对个数TN
    error_total = len(cos_wrong)  # 所有潜在非克隆对（应是非克隆对）个数TN+FP
    correct_recall = count / total
    precision = count / (error_total - error_count + count)  # error_total-error_count：潜在非克隆对中的克隆对数目
    F1 = 2 * precision * correct_recall / (precision + correct_recall)
    results = {'recall': correct_recall,
              'precision': precision,
              'F1': F1,
              'threshold': 0.5}
    for key, value in results.items():
        logger.info("threshold = 0.5 \t %s = %s", key, round(value, 4))

    x = []
    y = []
    for k in range(1, 100):

        count = 0
        error_count = 0
        threshold = k / 100
        for h in cos_right:
            if h[0] <= threshold:
                count += 1  # 实测克隆对个数TP
        total = len(cos_right)  # 所有潜在克隆对（应是克隆对）个数TP+FN
        for h in cos_wrong:
            if h[0] > threshold:
                error_count += 1  # 实测非克隆对个数TN
        error_total = len(cos_wrong)  # 所有潜在非克隆对（应是非克隆对）个数TN+FP
        correct_recall = count / total
        if error_total - error_count + count == 0:
            continue
        precision = count / (error_total - error_count + count)  # error_total-error_count：潜在非克隆对中的克隆对数目
        if precision + correct_recall == 0:
            continue
        F1 = 2 * precision * correct_recall / (precision + correct_recall)
        x.append(k)
        y.append(F1)

        if F1 > temp_best_f1:
            temp_best_f1 = F1
            temp_best_recall = correct_recall
            temp_best_precision = precision
            temp_count = count
            temp_error_count = error_count
            temp_error_total = error_total
            temp_total = total
            temp_best_threshold = threshold
    plt.plot(x, y, marker='o', linestyle='-', markersize=2)
    max_y = max(y)
    max_x = x[y.index(max_y)]
    plt.text(max_x, max_y, f'({max_x}, {max_y})', ha='right')
    plt.savefig(f'line_plot{idx}.png')
    print("eval_loss", temp_count, temp_error_count, temp_total, temp_error_total)
    result = {'recall': temp_best_recall,
              'precision': temp_best_precision,
              'F1': temp_best_f1,
              'threshold': temp_best_threshold}
    return result


def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = ExcelDataset(tokenizer, args, args.test_data_file)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    cos_right = []
    cos_wrong = []
    for batch in eval_dataloader:
        anchor = batch[0].to(args.device)
        positive = batch[1].to(args.device)
        negative = batch[2].to(args.device)
        with torch.no_grad():
            an_logits, po_logits, ne_logits = model(anchor, positive, negative)
        cos_r = (nn.CosineSimilarity(dim=1)(an_logits, po_logits) + 1) * 0.5
        cos_right += cos_r.tolist()
        cos_w = (nn.CosineSimilarity(dim=1)(an_logits, ne_logits) + 1) * 0.5
        cos_wrong += cos_w.tolist()
    temp_best_f1 = 0
    temp_best_recall = 0
    temp_best_precision = 0
    temp_count = 0
    temp_error_count = 0
    temp_error_total = 0
    temp_total = 0
    temp_best_threshold = 0

    count = 0
    error_count = 0
    if args.do_eval == False:
        best_threshold = 0.32
    logger.info("using eval_threshold: %s", best_threshold)
    for i in cos_right:
        if i[0] <= best_threshold:
            count += 1
    total = len(cos_right)
    for i in cos_wrong:
        if i[0] > best_threshold:
            error_count += 1
    error_total = len(cos_wrong)
    correct_recall = count / total
    precision = count / (error_total - error_count + count)
    F1 = 2 * precision * correct_recall / (precision + correct_recall)
    result = {'recall': correct_recall, 'precision': precision, 'F1': F1,
              'threshold': best_threshold}
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    for k in range(1, 100):
        count = 0
        error_count = 0
        threshold = k / 100
        for h in cos_right:
            if h[0] <= threshold:
                count += 1  # 实测克隆对个数TP
        total = len(cos_right)  # 所有潜在克隆对（应是克隆对）个数TP+FN
        for h in cos_wrong:
            if h[0] > threshold:
                error_count += 1  # 实测非克隆对个数TN
        error_total = len(cos_wrong)  # 所有潜在非克隆对（应是非克隆对）个数TN+FP
        correct_recall = count / total
        if error_total - error_count + count == 0:
            continue
        precision = count / (error_total - error_count + count)  # error_total-error_count：潜在非克隆对中的克隆对数目
        if precision + correct_recall == 0:
            continue
        F1 = 2 * precision * correct_recall / (precision + correct_recall)
        if F1 > temp_best_f1:
            temp_best_f1 = F1
            temp_best_recall = correct_recall
            temp_best_precision = precision
            temp_count = count
            temp_error_count = error_count
            temp_error_total = error_total
            temp_total = total
            temp_best_threshold = threshold
    print("eval_loss", temp_count, temp_error_count, temp_total, temp_error_total)
    result = {'recall': temp_best_recall,
              'precision': temp_best_precision,
              'F1': temp_best_f1,
              'threshold': temp_best_threshold}
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    logger.info("  " + "*" * 20)
    writeresult = 'recall: ' + str(round(temp_best_recall, 3)) + \
                  ' precision:' + str(round(temp_best_precision, 3)) + \
                  ' F1:' + str(round(temp_best_f1, 3)) + \
                  ' threshold:' + \
                  str(round(best_threshold, 1))
    f = open('result.txt', 'a+')
    f.write(writeresult)
    f.close()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # 添加参数
    # parser.add_argument("--train_data_file", default=None, type=str, required=True,
    #                     help="The input training data file (a text file).")
    # parser.add_argument("--train_data_file", default=None, type=str, required=True,
    #                     help="The input training data file (a text file).")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--cnn_size', type=int, default=1, help="For cnn size.")
    parser.add_argument('--filter_size', type=int, default=2, help="For cnn filter size.")
    
    parser.add_argument('--d_size', type=int, default=128, help="For cnn filter size.")
    parser.add_argument('--pkl_file', type=str, default='', help='for dataset path pkl file')
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    torch.cuda.set_device(0)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)  # min(512,510)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    # input_text =
    # input_ids =
    # outputs


    model = Model(model, config, tokenizer, args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = ExcelDataset(tokenizer, args, args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-acc/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            # model.load_state_dict(torch.load(output_dir))
            model.to(args.device)
            result = evaluate(args, model, tokenizer, 8)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key], 4)))
            
    if args.do_test and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-acc/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))                  
            model.to(args.device)
            test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()





