import csv
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader_Generation(dataset):
    dataset_source_folder_path = None
    dataset_source_file_name = None
    seq_len = 100

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading jokes generation data...')
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        jokes = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                jokes.append(row['Joke'].strip())

        corpus = '\n'.join(jokes)

        chars = sorted(set(corpus))
        char2idx = {'<PAD>': 0}
        for c in chars:
            char2idx[c] = len(char2idx)
        idx2char = {v: k for k, v in char2idx.items()}

        encoded = [char2idx[c] for c in corpus]

        X, y = [], []
        for i in range(0, len(encoded) - self.seq_len):
            X.append(encoded[i: i + self.seq_len])
            y.append(encoded[i + self.seq_len])

        print(f'  corpus length: {len(corpus)}, vocab: {len(char2idx)}, sequences: {len(X)}')
        return {
            'X': X, 'y': y,
            'char2idx': char2idx,
            'idx2char': idx2char,
            'vocab_size': len(char2idx),
        }
