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
# import ctree

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import json
from sklearn.metrics import recall_score, precision_score, f1_score
from tqdm import tqdm, trange
import multiprocessing

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
from c_cfg_3 import C_CFG
from parserTool.utils import remove_comments_and_docstrings
from parserTool.parse import Lang
import pickle

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def extract_pathtoken(source, path_sequence):
    seqtoken_out = []
    for path in path_sequence:
        seq_code = ''
        for line in path:
            if line in source:
                seq_code += source[line]
        seqtoken_out.append(seq_code)
        if len(seqtoken_out) > 10:
            break
    if len(path_sequence) == 0:
        seq_code = ''
        for i in source:
            seq_code += source[i]
        seqtoken_out.append(seq_code)
    # seqtoken_out = sorted(seqtoken_out, key=lambda i: len(i), reverse=False)
    return seqtoken_out

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str,
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
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    logger.info("start!!!")

    model.to(args.device)

    # 使用时恢复下行
    output = open('datasets/BCB/preprocess/path_embeddings_3_v2.pkl', 'wb')
    path_dict = {}
    state_dict = {}
    num_id = 0
    sum_ratio = 0
    sum_pr = 0
    num_path_dict = {}
    code_path = '../source_datasets/BCB/id2sourcecode'
    # code_path = '../source_datasets/BCB/test_code'
    file_list = os.listdir(code_path)

    for file in file_list:
        code = open(f"{code_path}/{file}", encoding='UTF-8').read()
        code = code.strip()
        num_id += 1
        if num_id%100==0:
            print(num_id, flush=True)
        clean_code, code_dict = remove_comments_and_docstrings(code, 'java')
        g = C_CFG()
        code_ast = ps.tree_sitter_ast(clean_code, Lang.JAVA)
        s_ast = g.parse_ast_file(code_ast.root_node)
        # ctree.ipython_show_ast(s_ast)
        num_path, cfg_allpath, _, ratio , path_real = g.get_allpath()
        # sum_ratio += ratio
        # sum_pr += path_real
        path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)

        all_seq_ids = []
        for seq in path_tokens1:
            seq = ' '.join(seq.split())
            seq_tokens = tokenizer.tokenize(seq)[:args.block_size - 2]
            seq_tokens = [tokenizer.cls_token] + seq_tokens + [tokenizer.sep_token]
            seq_ids = tokenizer.convert_tokens_to_ids(seq_tokens)
            padding_length = args.block_size - len(seq_ids)
            seq_ids += [tokenizer.pad_token_id] * padding_length
            all_seq_ids.append(seq_ids)
        # all_seq_ids = all_seq_ids[:args.filter_size]    # [3, 510]    [path_num, ids_len_of_path/token_len]
        all_seq_ids = all_seq_ids[:3]

        all_seq_ids = torch.tensor(all_seq_ids, dtype=torch.int64, device=args.device)
        # 计算路径表示E                                                    # [batch, path_num, ids_len_of_path/token_len]
        seq_embeds = model(all_seq_ids, attention_mask=all_seq_ids.ne(1))[0]  # [3, 510] -> [3, 510, 768]
        seq_embeds = seq_embeds[:, 0, :]  # [3, 510, 768] -> [3, 768]
        seq_embeds = seq_embeds.tolist()

        path_dict[file[:-5]] = seq_embeds, cfg_allpath
    print("test file finish...", flush=True)
    # print(sum_ratio/num_id, flush=True)
    # print(sum_pr/num_id, flush=True)

    # 使用时恢复下两行
    pickle.dump(path_dict, output)
    output.close()
    logger.info("finished!!!")


    # with open('test\messaget3.txt', 'w') as f:
    #     f.write(str(path_dict))
    #     f.write('\n')

    # embeds, cfg_allpath = path_dict['525']
    # embeds = torch.tensor(embeds,  dtype=torch.float32, device=args.device)


if __name__ == "__main__":
    main()





