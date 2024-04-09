import jsonlines

with open("../test_dataset/BCBcsv_onlyid/test/all/train_all_2.csv") as f1:
    for line in f1:
        dict_c = {}
        idx_list = line.strip().split(',')
        i = 1
        for idx in idx_list:
            code = open(f"../test_dataset/id2sourcecode/{idx}.java", encoding='UTF-8').read()
            dict_c[f'idx{i}'] = idx
            dict_c[f'code{i}'] = code
            i += 1
        with jsonlines.open('test/BCB/train_all_2.jsonl', mode='a') as writer:
            writer.write(dict_c)


# str = "9696025.java"
# print(str[:-6])

