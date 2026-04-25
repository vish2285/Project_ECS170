from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X = []
        y = []
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r') as f:
            for line in f:
                vals = line.strip().split(',')
                label = int(vals[0])
                # normalize pixels to [0, 1]
                pixels = [int(v) / 255.0 for v in vals[1:]]
                y.append(label)
                X.append(pixels)
        return {'X': X, 'y': y}
