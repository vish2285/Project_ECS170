import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from local_code.base_class.method import method


class ImageDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X          # list of numpy arrays (C, H, W)
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = torch.FloatTensor(self.X[idx])
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def _make_mnist_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),                                        # -> (64, 14, 14)
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),                                        # -> (128, 7, 7)
        nn.Flatten(),
        nn.Linear(128 * 7 * 7, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 10),
    )


def _make_orl_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),                                        # -> (32, 56, 46)
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),                                        # -> (64, 28, 23)
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),                                        # -> (128, 14, 11)
        nn.Flatten(),
        nn.Linear(128 * 14 * 11, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 40),
    )


def _make_cifar_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),                                        # -> (64, 16, 16)
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),                                        # -> (128, 8, 8)
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.MaxPool2d(2),                                        # -> (256, 4, 4)
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 10),
    )


AUGMENT = {
    'ORL': T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=8, translate=(0.05, 0.05)),
    ]),
    'MNIST': None,
    'CIFAR': T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ]),
}


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class Method_CNN(method, nn.Module):
    data = None
    dataset_type = None
    max_epoch = 50
    learning_rate = 1e-3
    batch_size = 64
    result_destination_folder_path = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.net = None
        self.device = _get_device()

    def _build_net(self):
        if self.dataset_type == 'MNIST':
            self.net = _make_mnist_cnn()
        elif self.dataset_type == 'ORL':
            self.net = _make_orl_cnn()
        elif self.dataset_type == 'CIFAR':
            self.net = _make_cifar_cnn()
        else:
            raise ValueError(f'Unknown dataset_type: {self.dataset_type}')
        self.net = self.net.to(self.device)

    def forward(self, x):
        return self.net(x)

    def train_model(self, X, y):
        print(f'  [device: {self.device}]', flush=True)
        train_transform = AUGMENT.get(self.dataset_type)
        dataset = ImageDataset(X, y, transform=train_transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        loss_fn = nn.CrossEntropyLoss()

        loss_history, acc_history = [], []

        for epoch in range(self.max_epoch):
            self.train()
            epoch_loss, correct, total = 0.0, 0, 0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.forward(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(yb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(yb)

            avg_loss = epoch_loss / total
            acc = correct / total
            loss_history.append(avg_loss)
            acc_history.append(acc)
            scheduler.step(avg_loss)

            if epoch % 5 == 0:
                print(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}', flush=True)

        self._save_curves(loss_history, acc_history)

    def _save_curves(self, loss_history, acc_history):
        epochs = range(1, len(loss_history) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, loss_history)
        ax1.set_title(f'{self.dataset_type} Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2.plot(epochs, acc_history)
        ax2.set_title(f'{self.dataset_type} Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        plt.tight_layout()
        path = self.result_destination_folder_path + f'{self.dataset_type}_convergence_curve.png'
        plt.savefig(path)
        plt.close()
        print(f'Convergence curve saved to {path}', flush=True)

    def test(self, X):
        self.eval()
        test_ds = ImageDataset(X, [0] * len(X))
        loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                preds.append(self.forward(xb).argmax(1).cpu())
        return torch.cat(preds).numpy()

    def run(self):
        self._build_net()
        print(f'method running on {self.dataset_type}...', flush=True)
        print('--start training...', flush=True)
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...', flush=True)
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
