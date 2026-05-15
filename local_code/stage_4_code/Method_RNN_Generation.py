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


class Method_RNN_Generation(method, nn.Module):
    data = None
    rnn_type = 'LSTM'       # 'RNN', 'LSTM', or 'GRU'
    vocab_size = None
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3
    max_epoch = 50
    learning_rate = 1e-3
    batch_size = 128
    result_destination_folder_path = None
    generate_length = 400
    temperature = 0.8
    start_text = 'why did the'

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.device = _get_device()
        self.embedding = None
        self.rnn = None
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
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
        self.to(self.device)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

    def train_model(self):
        X = torch.tensor(self.data['X'], dtype=torch.long)
        y = torch.tensor(self.data['y'], dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        loss_history = []
        print(f'  [{self.rnn_type} on {self.device}]', flush=True)

        for epoch in range(self.max_epoch):
            self.train()
            epoch_loss, total = 0.0, 0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                # forward: (batch, seq, vocab)
                logits, _ = self.forward(xb)
                # use final time step output to predict next char
                loss = loss_fn(logits[:, -1, :], yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item() * len(yb)
                total += len(yb)

            avg_loss = epoch_loss / total
            loss_history.append(avg_loss)
            scheduler.step(avg_loss)
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}', flush=True)

        self._save_curves(loss_history)

    def _save_curves(self, loss_history):
        epochs = range(1, len(loss_history) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, loss_history)
        ax.set_title(f'{self.rnn_type} Generation - Training Loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        plt.tight_layout()
        path = self.result_destination_folder_path + f'{self.rnn_type}_generation_convergence.png'
        plt.savefig(path)
        plt.close()
        print(f'Convergence curve saved to {path}', flush=True)

    def generate(self):
        self.eval()
        char2idx = self.data['char2idx']
        idx2char = self.data['idx2char']

        seed_chars = [c for c in self.start_text if c in char2idx]
        input_ids = [char2idx[c] for c in seed_chars]

        generated = list(seed_chars)
        hidden = None

        with torch.no_grad():
            # warm up hidden state on seed
            x = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            _, hidden = self.forward(x, hidden)

            # generate character by character
            last_char = input_ids[-1]
            for _ in range(self.generate_length):
                x = torch.tensor([[last_char]], dtype=torch.long).to(self.device)
                logits, hidden = self.forward(x, hidden)
                probs = torch.softmax(logits[0, 0] / self.temperature, dim=0)
                # mask <PAD> token
                probs[0] = 0.0
                probs = probs / probs.sum()
                last_char = torch.multinomial(probs, 1).item()
                generated.append(idx2char[last_char])

        return ''.join(generated)

    def run(self):
        self._build_net()
        print('--start training...', flush=True)
        self.train_model()
        print('--start generating...', flush=True)
        text = self.generate()
        print('\n--- Generated Text ---')
        print(text)
        print('--- End ---\n')

        out_path = self.result_destination_folder_path + f'{self.rnn_type}_generated_text.txt'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f'Start text: {self.start_text}\n\n')
            f.write(text)
        print(f'Generated text saved to {out_path}', flush=True)
        return {'generated_text': text, 'start_text': self.start_text}
