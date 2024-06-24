import jsonlines

with open("../source_datasets/BCB/BCBcsv_onlyid/type/all/type_train.csv") as f1:
    for line in f1:
        dict_c = {}
        idx_list = line.strip().split(',')
        i = 1
        for idx in idx_list:
            # code = open(f"../source_datasets/GCJ/googlejam4/{idx}", encoding='UTF-8').read()
            dict_c[f'idx{i}'] = idx
            # dict_c[f'code{i}'] = code
            i += 1
        with jsonlines.open('datasets/BCB/type/type_train.jsonl', mode='a') as writer:
            writer.write(dict_c)


# str = "9696025.java"
# print(str[:-6])

