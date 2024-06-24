import jsonlines

# with open("../source_datasets/GCJ/rand/all/short.csv") as f1:
#     for line in f1:
#         dict_c = {}
#         idx_list = line.strip().split(',')
#         i = 1
#         for idx in idx_list[:-1]:
#             # code = open(f"../source_datasets/GCJ/googlejam4/{idx}", encoding='UTF-8').read()
#             dict_c[f'idx{i}'] = idx
#             # dict_c[f'code{i}'] = code
#             i += 1
#         dict_c['label'] = int(idx_list[-1])
#         with jsonlines.open('datasets/GCJ/short.jsonl', mode='a') as writer:
#             writer.write(dict_c)

with open("../source_datasets/BCB/BCBcsv_onlyid/rand/all/eval_nobalance.csv") as f1:
    for line in f1:
        dict_c = {}
        idx_list = line.strip().split(',')
        i = 1
        for idx in idx_list[:-1]:
            # code = open(f"../source_datasets/BCB/id2sourcecode/{idx}.java", encoding='UTF-8').read()
            dict_c[f'idx{i}'] = idx
            # dict_c[f'code{i}'] = code
            i += 1
        dict_c['label'] = int(idx_list[-1])
        with jsonlines.open('datasets/BCB/eval_nobalance.jsonl', mode='a') as writer:
            writer.write(dict_c)


# str = "9696025.java"
# print(str[:-6])

