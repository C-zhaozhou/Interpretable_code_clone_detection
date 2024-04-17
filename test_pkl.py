# import pickle
# pkl_file = open("short_3path_cdata_nobalance.pkl", 'rb')
# path_dict = pickle.load(pkl_file)
# print(path_dict)

import os
import sys

import parserTool.parse as ps
from c_cfg import C_CFG
from parserTool.utils import remove_comments_and_docstrings
from parserTool.parse import Lang
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import logging
import numpy as np


def extract_pathtoken(source, path_sequence):
    seqtoken_out = []
    for path in path_sequence:
        seq_code = ''
        for line in path:
            if (line in source):
                seq_code += source[line]
        seqtoken_out.append(seq_code)
        if len(seqtoken_out) > 10:
            break
    if len(path_sequence) == 0:
        seq_code = ''
        for i in source:
            seq_code += source[i]
        seqtoken_out.append(seq_code)
    seqtoken_out = sorted(seqtoken_out, key=lambda i: len(i), reverse=False)
    return seqtoken_out


def main():
    output = open('datasets/test.pkl', 'wb')
    path_dict = {}
    state_dict = {}
    num_id = 0
    sum_ratio = 0
    num_path_dict = {}

    with open("../test_dataset/BCBcsv_onlyid/clone_test.csv") as f1:
        for line in f1:
            group = []
            idx_list = line.strip().split(',')
            for idx in idx_list:
                code = open(f"../test_dataset/id2sourcecode/{idx}.java", encoding='UTF-8').read()
                # group.append(code)

    # with open("../dataset/cdata/nobalance/test.jsonl") as f:
    #     for i in [1, 10]:
    #         line = f.readline()
    #
    #     # for line in f:
    #         num_id += 1
    #         if num_id % 100 == 0:
    #             print(num_id, flush=True)
    #
    #         js = json.loads(line.strip())

                clean_code, code_dict = remove_comments_and_docstrings(code, 'java')
                print(clean_code)
                print(code_dict)
                g = C_CFG()
                code_ast = ps.tree_sitter_ast(clean_code, Lang.JAVA)
                # print(code_ast, type(code_ast))
                # print(code_ast.root_node, type(code_ast.root_node))

                # 使用Lang.C和Lang.JAVA生成的根节点的种类不同，导致parse的时候出现问题
                s_ast = g.parse_ast_file(code_ast.root_node)

                pos = nx.spring_layout(g.G)
                nx.draw(g.G, with_labels=True, font_weight='bold')

                plt.show()

                print("****************")
                print(code_ast)
                print(s_ast)
                print("****************")
                num_path, cfg_allpath, _, ratio = g.get_allpath()
                print(f"ratio = {ratio}")
                print(cfg_allpath)
                path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
                sum_ratio += ratio
                path_dict[idx] = path_tokens1, cfg_allpath
                # print("num_paths:", num_path)
    print("train file finish...", flush=True)
    print(sum_ratio/num_id, flush=True)
    # Pickle dictionary using protocol 0.
    pickle.dump(path_dict, output)
    output.close()

if __name__=="__main__":
    main()

