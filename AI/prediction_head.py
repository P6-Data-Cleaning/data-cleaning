import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    def __init__(self, fusion_dim, output_dim=2):
        """
        Initialize the prediction head for coordinate prediction
        
        Args:
            fusion_dim (int): Dimension of the fused input features
            output_dim (int): Dimension of the output predictions (default=2 for lat/lon)
        """
        super().__init__()
        
        # Create a small MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(fusion_dim // 2, output_dim)
        )
        
    def forward(self, fused_features):
        """
        Predict coordinates from fused features
        
        Args:
            fused_features (torch.Tensor): Fused temporal and spatial features 
                                         shape: (batch_size, fusion_dim)
                                         
        Returns:
            torch.Tensor: Predicted coordinates, shape: (batch_size, output_dim)
        """
        return self.mlp(fused_features) 