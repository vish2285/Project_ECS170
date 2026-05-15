import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from local_code.base_class.method import method


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class Method_RNN_Classification(method, nn.Module):
    data = None
    rnn_type = 'LSTM'       # 'RNN', 'LSTM', or 'GRU'
    vocab_size = 10002
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3
    max_epoch = 10
    learning_rate = 1e-3
    batch_size = 64
    result_destination_folder_path = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.device = _get_device()
        self.embedding = None
        self.rnn = None
        self.drop = None
        self.fc = None

    def _build_net(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[self.rnn_type]
        rnn_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            self.embed_dim, self.hidden_dim,
            num_layers=self.num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(self.hidden_dim, 2)
        self.to(self.device)

    def forward(self, x):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb)
        # Take the last layer's final hidden state
        if self.rnn_type == 'LSTM':
            h_n = hidden[0]
        else:
            h_n = hidden
        last = h_n[-1]  # (batch, hidden_dim)
        return self.fc(self.drop(last))

    def train_model(self):
        X = torch.tensor(self.data['train']['X'], dtype=torch.long)
        y = torch.tensor(self.data['train']['y'], dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
        loss_fn = nn.CrossEntropyLoss()

        loss_history, acc_history = [], []
        print(f'  [{self.rnn_type} on {self.device}]', flush=True)

        for epoch in range(self.max_epoch):
            self.train()
            epoch_loss, correct, total = 0.0, 0, 0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)

            avg_loss = epoch_loss / total
            acc = correct / total
            loss_history.append(avg_loss)
            acc_history.append(acc)
            scheduler.step(avg_loss)
            print(f'Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}', flush=True)

        self._save_curves(loss_history, acc_history)

    def _save_curves(self, loss_history, acc_history):
        epochs = range(1, len(loss_history) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, loss_history)
        ax1.set_title(f'{self.rnn_type} Classification - Loss')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax2.plot(epochs, acc_history)
        ax2.set_title(f'{self.rnn_type} Classification - Accuracy')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
        plt.tight_layout()
        path = self.result_destination_folder_path + f'{self.rnn_type}_classification_convergence.png'
        plt.savefig(path)
        plt.close()
        print(f'Convergence curve saved to {path}', flush=True)

    def test_model(self):
        self.eval()
        X = torch.tensor(self.data['test']['X'], dtype=torch.long)
        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                preds.append(self.forward(xb).argmax(1).cpu())
        return torch.cat(preds).numpy().tolist()

    def run(self):
        self._build_net()
        print('--start training...', flush=True)
        self.train_model()
        print('--start testing...', flush=True)
        pred_y = self.test_model()
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
