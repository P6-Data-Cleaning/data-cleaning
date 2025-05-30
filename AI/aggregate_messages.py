import torch_scatter

def aggregate_neighbor_messages(hidden_states, edge_index):
    """ hidden_states: (num_ships, hidden_dim).

    edge_index: shape [2, num_edges], where source nodes send messages to target nodes.

    Use scatter_ (or torch_scatter) to perform aggregation.
    """

    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    messages = hidden_states[source_nodes] # Get neighbor messages
    # For each target, compute mean of messages
    aggregated = torch_scatter.scatter_mean(messages, target_nodes, dim=0, dim_size=hidden_states.size(0))
    return aggregated