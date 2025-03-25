import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, hidden_dim, fusion_dim):
        super().__init__()
        self.fc = nn.Linear(2 * hidden_dim, fusion_dim)
        self.relu = nn.ReLU()
        
    def forward(self, self_state, neighbor_message):
        """
        Fuses temporal and spatial information for each ship
        
        Args:
            self_state: Ship's own temporal hidden state (batch_size, hidden_dim)
            neighbor_message: Aggregated neighbor messages (batch_size, hidden_dim)
            
        Returns:
            Fused representation (batch_size, fusion_dim)
        """
        # Concatenate temporal and spatial features
        combined = torch.cat([self_state, neighbor_message], dim=1)
        
        # Project to fusion dimension
        fused = self.fc(combined)
        fused = self.relu(fused)
        
        return fused
