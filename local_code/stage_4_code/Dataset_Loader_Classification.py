import os
import re
from collections import Counter
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader_Classification(dataset):
    dataset_source_folder_path = None
    max_vocab_size = 10000
    max_seq_len = 200

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _clean(self, text):
        text = text.lower()
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()

    def _load_split(self, split):
        texts, labels = [], []
        for label_str, label_int in [('pos', 1), ('neg', 0)]:
            folder = os.path.join(self.dataset_source_folder_path, split, label_str)
            for fname in sorted(os.listdir(folder)):
                if not fname.endswith('.txt'):
                    continue
                with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                    texts.append(self._clean(f.read()))
                labels.append(label_int)
        return texts, labels

    def load(self):
        print('loading IMDb classification data...')
        train_texts, train_labels = self._load_split('train')
        test_texts, test_labels = self._load_split('test')

        # Build vocab from train only
        counter = Counter(w for tokens in train_texts for w in tokens)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in counter.most_common(self.max_vocab_size - 2):
            vocab[word] = len(vocab)

        def numericalize(tokens):
            ids = [vocab.get(w, 1) for w in tokens[:self.max_seq_len]]
            ids += [0] * (self.max_seq_len - len(ids))
            return ids

        X_train = [numericalize(t) for t in train_texts]
        X_test = [numericalize(t) for t in test_texts]

        print(f'  train: {len(X_train)}, test: {len(X_test)}, vocab: {len(vocab)}')
        return {
            'X_train': X_train, 'y_train': train_labels,
            'X_test': X_test, 'y_test': test_labels,
            'vocab_size': len(vocab),
        }
