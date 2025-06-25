import numpy as np
import networkx as nx
import time
from datetime import datetime
import pickle
import os
import json
import threading
from contextlib import contextmanager

class VirtualEdgeCache:
    def __init__(self, cache_file=None):
        self.cache_file = cache_file

    def get_virtual_edges(self, nodes, edges):
        """Get virtual edges for a shape, either from cache or by creating new ones"""
        virtual_edges = create_hybrid_virtual_edges(nodes, edges)
      
        return virtual_edges   

def create_hybrid_virtual_edges(nodes, edges, percentage=0.1333):
    """Create virtual edges using simple random approach."""
    print(f"\nStarting random edge creation at {datetime.now()}")
    start_time = time.time()
    
    total_allowed_edges = int(len(edges) * percentage)
    print(f"Total edge limit: {total_allowed_edges}")
    
    # Initialize empty list for virtual edges
    virtual_edges = []
    n_nodes = len(nodes)
    
    # Create set of existing edges for faster lookup
    existing_edges = set(tuple(sorted(edge)) for edge in edges.keys())
    
    # Generate random edges until we reach the limit
    while len(virtual_edges) < total_allowed_edges:
        # Randomly select two different nodes
        node1, node2 = np.random.choice(n_nodes, size=2, replace=False)
        edge = tuple(sorted([node1, node2]))
        
        # Add edge if it doesn't already exist
        if edge not in existing_edges and edge not in virtual_edges:
            virtual_edges.append(edge)
    
    print(f"Random edges complete: {len(virtual_edges)} edges added")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
    
    return virtual_edges


def add_virtual_edges(edges, virtual_edges, transformed_coords, use_axial_stress=False):
    """Add virtual edges to the existing edges dictionary."""
    # Ensure all existing edges have the correct number of features
    for edge in edges:
        if len(edges[edge]) < 5:
            edges[edge].append(0)  # Add virtual edge indicator (0 for real edges)
        if len(edges[edge]) < 6 and use_axial_stress:
            edges[edge].append(0)  # Add axial stress feature if needed
    
    # Add virtual edges with appropriate features
    for edge in virtual_edges:
        idx1, idx2 = edge
        
        pos1 = transformed_coords[idx1]
        pos2 = transformed_coords[idx2]
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance = np.sqrt(dx**2 + dy**2)
        direction = np.array([dx, dy]) / distance
        
        
        if use_axial_stress:
            edges[edge] = [0, distance/1000, direction[0], direction[1], 0, 1]  # 1 indicates virtual edge
        else:
            edges[edge] = [0, distance/1000, direction[0], direction[1], 1]  # 1 indicates virtual edge
    
    return edges

def create_super_node(nodes, edges, feature_size):
    """Create a super node at the center of the graph and connect it to all nodes."""
    # Add node type feature to existing nodes if needed
    node_features = []
    for node_feature in nodes:
        if len(node_feature) < feature_size + 1:  # If node type feature doesn't exist
            new_feature = list(node_feature) + [0]  # Add 0 for real node
        else:
            new_feature = list(node_feature)
        node_features.append(new_feature)
    
    # Create super node features
    super_node_features = [0] * (feature_size + 1)  # +1 for node type feature

    # # Calculate centroid from node coordinates (first 2 or 3 features)
    # coords = np.array([node[:2] for node in nodes])  # Always take first 2 coordinates
    # centroid = np.mean(coords, axis=0)
    
    # # Set centroid coordinates as first features of supernode
    # super_node_features[0] = centroid[0]
    # super_node_features[1] = centroid[1]

    super_node_features[-1] = 1  # Set virtual node indicator to 1
    
    # Create edges from super node to all other nodes
    num_nodes = len(nodes)
    super_node_idx = num_nodes  # Index for the super node
    super_node_edges = []
    
    for i in range(num_nodes):
        super_node_edges.append((super_node_idx, i))
    
    return node_features, super_node_features, super_node_edges