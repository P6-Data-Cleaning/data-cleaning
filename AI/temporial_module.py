import torch.nn as nn

class ship_temporial_updater(nn.Module):
    def init(self, input_dim, hidden_dim):
        super().init()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    # Foward propagation
    def forward(self, x):
        embedded = self.embedding(x) # shape: (batch_size, sequence_length, hidden_dim)
        output, hidden = self.gru(embedded)
        # output: per time-step state, hidden: last time-step state for each sequence
        return output, hidden