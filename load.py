import pickle
import json

fr = open(r"datasets\path_embeddings.pkl", "rb")
result = pickle.load(fr)


with open('datasets\messaget2.txt', 'w') as f:
    json_str = json.dumps(result, indent=0)
    f.write(json_str)
    f.write('\n')
# print(result)

# fr = open(r"test\messaget3.txt", "r")
# lines = fr.read()
# print(lines)
# dict_ = eval(lines)
# print(dict_)
