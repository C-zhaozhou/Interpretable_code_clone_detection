import jsonlines

with open("../test_dataset/BCBcsv_onlyid/clone_eval.csv") as f1:
    for line in f1:
        dict_c = {}
        idx_list = line.strip().split(',')
        i = 1
        for idx in idx_list[:-1]:
            code = open(f"../test_dataset/id2sourcecode/{idx}.java", encoding='UTF-8').read()
            dict_c[f'idx{i}'] = idx
            dict_c[f'code{i}'] = code
            i += 1
        dict_c['label'] = int(idx_list[-1])
        with jsonlines.open('test/eval_test.jsonl', mode='a') as writer:
            writer.write(dict_c)


# str = "9696025.java"
# print(str[:-6])

