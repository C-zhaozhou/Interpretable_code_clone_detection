import parserTool.parse as ps
from c_cfg import C_CFG
from parserTool.utils import remove_comments_and_docstrings
from parserTool.parse import Lang
import pickle
import json
import sys
import logging
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


# def convert_examples_to_features(js, tokenizer, path_dict, args):
#     clean_code, code_dict = remove_comments_and_docstrings(js['func'], 'c')
#
#     # source
#     code = ' '.join(clean_code.split())
#     code_tokens = tokenizer.tokenize(code)[:400 - 2]
#     source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
#     source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
#     padding_length = 400 - len(source_ids)
#     source_ids += [tokenizer.pad_token_id] * padding_length
#
#     if js['idx'] in path_dict:
#         path_tokens1, cfg_allpath = path_dict[js['idx']]
#     else:
#         clean_code, code_dict = remove_comments_and_docstrings(js['func'], 'c')
#         g = C_CFG()
#         code_ast = ps.tree_sitter_ast(clean_code, Lang.C)
#         s_ast = g.parse_ast_file(code_ast.root_node)
#         num_path, cfg_allpath = g.get_allpath()
#         path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
#
#     all_seq_ids = []
#     for seq in path_tokens1:
#         seq_tokens = tokenizer.tokenize(seq)[:400 - 2]
#         seq_tokens = [tokenizer.cls_token] + seq_tokens + [tokenizer.sep_token]
#         seq_ids = tokenizer.convert_tokens_to_ids(seq_tokens)
#         padding_length = 400 - len(seq_ids)
#         seq_ids += [tokenizer.pad_token_id] * padding_length
#         all_seq_ids.append(seq_ids)
#
#     if len(all_seq_ids) < 3:
#         for i in range(3 - len(all_seq_ids)):
#             all_seq_ids.append(source_ids)
#     else:
#         all_seq_ids = all_seq_ids[:3]
#     return CloneFeatures(source_tokens, source_ids, all_seq_ids, js['idx'], js['target'])

# def convert(idx):


# code_list = [source_code1,source_code2,source_code3]
# clean_code = [clean_code1,clean_code2,clean_code3]
# code_dict = [dic_code1,dic_code2,dic_code3] 代码段与代码行数（index）的对应关系


class CloneFeatures(object):
    def __init__(self,
                 an_all_seq_ids,
                 po_all_seq_ids,
                 ne_all_seq_ids,
    ):
        self.an_all_seq_ids = an_all_seq_ids
        self.po_all_seq_ids = po_all_seq_ids
        self.ne_all_seq_ids = ne_all_seq_ids


def convert_examples_to_features_clone(code_list, tokenizer):
    codes_paths = []  # 3*3*……
    for i, snippet in enumerate(code_list):
        clean_code, code_dict = remove_comments_and_docstrings(snippet, 'c')

        # source
        pre_code = ' '.join(clean_code.split())
        code_tokens = tokenizer.tokenize(pre_code)[:400 - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 400 - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        # paths
        g = C_CFG()
        code_ast = ps.tree_sitter_ast(clean_code, Lang.JAVA)
        s_ast = g.parse_ast_file(code_ast.root_node)
        num_path, cfg_allpath, _, _ = g.get_allpath()
        path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)

        all_seq_ids = []
        for seq in path_tokens1:
            seq_tokens = tokenizer.tokenize(seq)[:400 - 2]
            seq_tokens = [tokenizer.cls_token] + seq_tokens + [tokenizer.sep_token]
            seq_ids = tokenizer.convert_tokens_to_ids(seq_tokens)
            padding_length = 400 - len(seq_ids)
            seq_ids += [tokenizer.pad_token_id] * padding_length
            all_seq_ids.append(seq_ids)

        if len(all_seq_ids) < 3:
            for j in range(3 - len(all_seq_ids)):
                all_seq_ids.append(source_ids)
        else:
            all_seq_ids = all_seq_ids[:3]
        codes_paths.append(all_seq_ids)
    return CloneFeatures(codes_paths[0], codes_paths[1], codes_paths[2])


with open("../test_dataset/BCBcsv_onlyid/clone_test.csv") as f1:
    for line in f1:
        group = []
        idx_list = line.strip().split(',')
        for idx in idx_list:
            code = open(f"../test_dataset/id2sourcecode/{idx}.java", encoding='UTF-8').read()
            group.append(code)
        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
        tokenizer = tokenizer_class.from_pretrained('../models/codebert')
        convert_examples_to_features_clone(group, tokenizer)




        # an_code = convert(an)
    # with open("../dataset/cdata/nobalance/train_mini.jsonl") as f:
    #     for i in [1]:
    #         line = f.readline()
    #         js = json.loads(line.strip())
    #
    #         clean_code2, code_dict2 = remove_comments_and_docstrings(js['func'], 'c')
    #         # source
    #         code2 = ' '.join(clean_code2.split())
    #
    #         print(code2)
