import torch
from torch_geometric.nn import radius_graph

def get_neighbor_edges(positions, radius):
    # positions: (num_ships, 2) tensor (latitude, longitude)
    # radius: spatial threshold (make sure units are consistent)
    edge_index = radius_graph(positions, r=radius, loop=False)
    return edge_index


def filter_edges_by_dynamic(edge_index, headings, heading_threshold):
    """
    Filter edges based on heading similarity between vessels.
    
    Args:
        edge_index: (2, num_edges) tensor containing source and target node indices
        headings: (num_ships,) tensor of vessel headings in degrees
        heading_threshold: maximum allowed difference in heading
        
    Returns:
        filtered_edge_index: (2, num_filtered_edges) tensor with edges that meet heading criteria
    """
    # Extract source and target node indices
    src, dst = edge_index[0], edge_index[1]

    # Get headings for source and target nodes
    src_headings = headings[src]
    dst_headings = headings[dst]

    # Compute absolute difference in headings
    heading_diff = torch.abs(src_headings - dst_headings)

    # Handle circular nature of headings (e.g., 359° vs 1° should be 2°, not 358°)
    # For headings in degrees (0-360)
    heading_diff = torch.min(heading_diff, 360 - heading_diff)

    # Create mask for edges that meet heading criteria
    mask = heading_diff < heading_threshold

    # Filter edge index based on mask
    filtered_edge_index = edge_index[:, mask]

    return filtered_edge_index