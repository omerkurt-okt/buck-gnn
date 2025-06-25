import matplotlib.pyplot as plt
import numpy as np
import os as os

def visualize_graph(node_features, edge_index, edge_features, op2_file=None):
    """
    Visualize a graph with labels for randomly selected nodes and edges.
    
    Parameters:
    -----------
    node_features : numpy.ndarray
        Features of each node
    edge_index : numpy.ndarray
        Edge connectivity information
    edge_features : numpy.ndarray
        Features of each edge
    op2_file : str, optional
        Name of the OP2 file for the title
    """
    # Create a plot
    plt.figure(figsize=(12, 10))
    
    # Plot edges
    for i in range(edge_index.shape[1]):
        start_idx = edge_index[0, i]
        end_idx = edge_index[1, i]
        start_pos = node_features[start_idx, :2]
        end_pos = node_features[end_idx, :2]
        
        # Color based on edge type
        if edge_features[i, -1] == 1:  # Virtual edge
            color = 'green'
            alpha = 0.5
            linewidth = 1
        else:
            color = 'red' if edge_features[i, 0] > 0.5 else 'yellow'
            alpha = 0.5
            linewidth = 2 if color == 'red' else 1
            
        plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                color=color, alpha=alpha, linewidth=linewidth)
    
    # Plot nodes
    plt.scatter(node_features[:, 0], node_features[:, 1], c='blue', s=50)
    
    # Randomly select 2 nodes for labeling
    selected_nodes = np.random.choice(node_features.shape[0], size=2, replace=False)
    for node in selected_nodes:
        x, y = node_features[node, :2]
        features = node_features[node]
        label = f"Node {node}\n"
        for i, feat in enumerate(features):
            label += f"F{i+1}: {feat:.4f}\n"
        plt.text(x, y, label, fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    # Randomly select 2 edges for labeling
    num_edges = edge_index.shape[1]
    selected_edges = np.random.choice(num_edges, size=2, replace=False)
    for edge_idx in selected_edges:
        start_idx = edge_index[0, edge_idx]
        end_idx = edge_index[1, edge_idx]
        x = (node_features[start_idx, 0] + node_features[end_idx, 0]) / 2
        y = (node_features[start_idx, 1] + node_features[end_idx, 1]) / 2
        features = edge_features[edge_idx]
        label = f"Edge {edge_idx}\n({start_idx}-{end_idx})\n"
        for i, feat in enumerate(features):
            label += f"F{i+1}: {feat:.4f}\n"
        plt.text(x, y, label, fontsize=8, bbox=dict(facecolor='lightyellow', alpha=0.7))
    
    # Update legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=2, label='Stiffener (CBAR)'),
        plt.Line2D([0], [0], color='yellow', label='Shell Element'),
        plt.Line2D([0], [0], color='green', label='Virtual Edge'),
        plt.Line2D([0], [0], marker='o', color='blue', label='Nodes',
                  markersize=10, linestyle='None')
    ]
    plt.legend(handles=legend_elements)
    
    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    if op2_file:
        op2_name = os.path.basename(op2_file)
        plt.title(f'Graph Visualization - {op2_name}')
    else:
        plt.title('Graph Visualization')
    
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()