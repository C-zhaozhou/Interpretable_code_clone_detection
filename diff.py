import difflib

def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        content1 = f1.readlines()
        content2 = f2.readlines()

    d = difflib.Differ()
    differences = list(d.compare(content1, content2))
    for line in differences:
        print(line)

if __name__ == '__main__':
    compare_files("run.py","run_mlp.py")


