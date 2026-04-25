from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 100
    learning_rate = 1e-3
    batch_size = 256
    # path to save the convergence curve image
    result_destination_folder_path = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

    def train_model(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        loss_fn = nn.CrossEntropyLoss()

        X_t = torch.FloatTensor(np.array(X))
        y_t = torch.LongTensor(np.array(y))
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        loss_history = []
        acc_history = []

        for epoch in range(self.max_epoch):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in loader:
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

            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}')

        self._save_curves(loss_history, acc_history)

    def _save_curves(self, loss_history, acc_history):
        epochs = range(1, len(loss_history) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(epochs, acc_history)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        plt.tight_layout()
        path = self.result_destination_folder_path + 'convergence_curve.png'
        plt.savefig(path)
        plt.close()
        print(f'Convergence curve saved to {path}')

    def test(self, X):
        self.eval()
        with torch.no_grad():
            out = self.forward(torch.FloatTensor(np.array(X)))
        return out.argmax(1).numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
