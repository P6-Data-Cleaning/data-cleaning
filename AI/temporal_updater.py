import torch
import torch.nn as nn

class ShipTemporalUpdater(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize a temporal processing module for ship trajectory data
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden state
        """
        super().__init__()
        
        # GRU for temporal processing
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x_seq):
        """
        Process a sequence of ship features
        
        Args:
            x_seq (torch.Tensor): Sequence of features (batch_size, seq_len, input_dim)
            
        Returns:
            tuple: (
                output: Sequence of hidden states (batch_size, seq_len, hidden_dim),
                h_n: Final hidden state (1, batch_size, hidden_dim)
            )
        """
        # Process sequence through GRU
        output, h_n = self.gru(x_seq)
        
        return output, h_n 