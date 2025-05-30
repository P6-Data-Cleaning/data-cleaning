import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import radius_graph

from temporal_updater import ShipTemporalUpdater
from fusion_module import FusionModule
from prediction_head import PredictionHead

class DynamicGraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, fusion_dim, max_radius=1.0):
        """
        Initialize the complete dynamic graph model for ship trajectory prediction
        
        Args:
            input_dim (int): Dimension of input features per ship
            hidden_dim (int): Hidden dimension for temporal and spatial processing
            fusion_dim (int): Dimension for fused representations
            max_radius (float): Maximum radius for considering neighbor ships
        """
        super().__init__()
        
        # Core components
        self.temporal_updater = ShipTemporalUpdater(input_dim, hidden_dim)
        self.fusion_module = FusionModule(hidden_dim, fusion_dim)
        self.prediction_head = PredictionHead(fusion_dim)
        
        # Model parameters
        self.max_radius = max_radius
        self.hidden_dim = hidden_dim
        
    def _filter_edges_by_heading(self, edge_index, headings, threshold):
        """
        Filter edges based on heading difference between ships
        
        Args:
            edge_index (torch.Tensor): Graph connectivity (2, num_edges)
            headings (torch.Tensor): Ship headings (batch_size,)
            threshold (float): Maximum allowed heading difference
            
        Returns:
            torch.Tensor: Filtered edge_index
        """
        src, dst = edge_index
        heading_diff = torch.abs(headings[src] - headings[dst])
        # Normalize heading difference to [0, 180]
        heading_diff = torch.min(heading_diff, 360 - heading_diff)
        mask = heading_diff <= threshold
        return edge_index[:, mask]
    
    def _aggregate_neighbor_messages(self, hidden_states, edge_index):
        """
        Aggregate messages from neighbor ships
        
        Args:
            hidden_states (torch.Tensor): Hidden states for all ships (batch_size, hidden_dim)
            edge_index (torch.Tensor): Graph connectivity (2, num_edges)
            
        Returns:
            torch.Tensor: Aggregated neighbor messages (batch_size, hidden_dim)
        """
        src, dst = edge_index
        # Simple mean aggregation of neighbor states
        neighbor_states = hidden_states[src]
        
        # Initialize output tensor with zeros
        batch_size = hidden_states.size(0)
        aggregated = torch.zeros(batch_size, self.hidden_dim, device=hidden_states.device)
        
        # Perform scatter_mean to aggregate messages
        aggregated.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.hidden_dim), neighbor_states)
        # Normalize by neighbor count
        neighbor_counts = torch.zeros(batch_size, device=hidden_states.device)
        neighbor_counts.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        neighbor_counts = torch.clamp(neighbor_counts, min=1.0).unsqueeze(1)
        aggregated = aggregated / neighbor_counts
        
        return aggregated
        
    def forward(self, x_seq, pos_seq, headings=None, heading_threshold=None):
        """
        Process a batch of ship sequences
        
        Args:
            x_seq (torch.Tensor): Sequence of features per ship (batch_size, seq_len, input_dim)
            pos_seq (torch.Tensor): Positions over time (batch_size, seq_len, 2)
            headings (torch.Tensor, optional): Ship headings (batch_size,)
            heading_threshold (float, optional): Maximum allowed heading difference
            
        Returns:
            dict: {
                'predictions': Predicted next positions,
                'temporal_states': Hidden states from temporal processing,
                'fused_states': States after fusion
            }
        """
        # Process sequence through temporal updater
        temporal_out, hidden = self.temporal_updater(x_seq)
        
        # Use the last time step's state as current state
        current_state = temporal_out[:, -1, :]  # (batch_size, hidden_dim)
        current_positions = pos_seq[:, -1, :]   # (batch_size, 2)
        
        # Compute neighbor edges using current positions
        edge_index = radius_graph(current_positions, 
                                r=self.max_radius,
                                batch=None,
                                loop=False)
        
        # Optionally filter edges by heading
        if headings is not None and heading_threshold is not None:
            edge_index = self._filter_edges_by_heading(edge_index, headings, heading_threshold)
        
        # Aggregate neighbor messages
        neighbor_message = self._aggregate_neighbor_messages(current_state, edge_index)
        
        # Fuse self state with neighbor message
        fused = self.fusion_module(current_state, neighbor_message)
        
        # Generate predictions
        predictions = self.prediction_head(fused)
        
        return {
            'predictions': predictions,
            'temporal_states': temporal_out,
            'fused_states': fused
        } 