import numpy as np

def read_data(filename):
    with open(filename, 'r', encoding="utf8") as f:
        data = f.read()[:1000]
    print(data)
    return data

