import pickle
import numpy as np
from local_code.base_class.dataset import dataset

# CIFAR-10 per-channel normalization constants
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
CIFAR_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None  # pickle file name: 'ORL', 'MNIST', or 'CIFAR'
    dataset_type = None              # 'ORL', 'MNIST', or 'CIFAR'
    split = 'train'                  # 'train' or 'test'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print(f'loading {self.dataset_type} {self.split} data...')
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(path, 'rb') as f:
            raw = pickle.load(f)

        instances = raw[self.split]
        imgs_raw = np.array([inst['image'] for inst in instances], dtype=np.float32)
        labels = [inst['label'] for inst in instances]

        if self.dataset_type == 'ORL':
            # (N, 112, 92, 3) -> use R channel -> (N, 1, 112, 92), normalize to [0, 1]
            imgs = imgs_raw[:, :, :, 0] / 255.0
            imgs = imgs[:, np.newaxis, :, :]
            labels = [l - 1 for l in labels]  # shift 1-40 to 0-39

        elif self.dataset_type == 'MNIST':
            # (N, 28, 28) -> (N, 1, 28, 28), normalize to [0, 1]
            imgs = imgs_raw / 255.0
            imgs = imgs[:, np.newaxis, :, :]

        elif self.dataset_type == 'CIFAR':
            # (N, 32, 32, 3) -> (N, 3, 32, 32), normalize per channel
            imgs = imgs_raw / 255.0
            imgs = imgs.transpose(0, 3, 1, 2)  # (N, 3, 32, 32)
            imgs = (imgs - CIFAR_MEAN) / CIFAR_STD

        else:
            imgs = imgs_raw

        return {'X': list(imgs), 'y': labels}
