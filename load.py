import pickle
import json

fr = open(r"test\test.pkl", "rb")
result = pickle.load(fr)


with open('test\messaget.txt', 'w') as f:
    json_str = json.dumps(result, indent=0)
    f.write(json_str)
    f.write('\n')
# print(result)

