import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import os
import time
from datetime import datetime
from scipy import spatial
from scipy.spatial import ConvexHull
def create_quad_mesh(n_points_per_side=15):
    """Create a square domain with quadrilateral mesh."""
    # Create grid points
    x = np.linspace(-1, 1, n_points_per_side)
    y = np.linspace(-1, 1, n_points_per_side)
    X, Y = np.meshgrid(x, y)
    
    # Create nodes
    nodes = np.column_stack((X.flatten(), Y.flatten()))
    
    # Create edges
    edges = {}
    for i in range(n_points_per_side):
        for j in range(n_points_per_side):
            current_node = i * n_points_per_side + j
            
            # Connect to right neighbor
            if j < n_points_per_side - 1:
                right_node = current_node + 1
                edge = tuple(sorted([current_node, right_node]))
                edges[edge] = [0.01, 2.0/n_points_per_side, 2.0/n_points_per_side, 0]
            
            # Connect to bottom neighbor
            if i < n_points_per_side - 1:
                bottom_node = current_node + n_points_per_side
                edge = tuple(sorted([current_node, bottom_node]))
                edges[edge] = [0.01, 2.0/n_points_per_side, 0, 2.0/n_points_per_side]
            
            # Connect to diagonal
            if i < n_points_per_side - 1 and j < n_points_per_side - 1:
                diagonal_node = current_node + n_points_per_side + 1
                edge = tuple(sorted([current_node, diagonal_node]))
                edges[edge] = [0.01, 2.0*np.sqrt(2)/n_points_per_side, 
                             2.0/n_points_per_side, 2.0/n_points_per_side]
    
    return nodes, edges

def calculate_centrality_score(centrality, node1, node2):
    """Calculate combined centrality score for a node pair."""
    
    return centrality[node1] + centrality[node2]

def find_longest_path_edge(G, edges, nodes):
    """Find the edge that connects nodes with the longest shortest path,
    considering centrality for ties."""
    longest_paths = []
    max_length = 0
    
    # Calculate all shortest paths
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if (i, j) not in edges and (j, i) not in edges:
                try:
                    path_length = nx.shortest_path_length(G, i, j)
                    if path_length > max_length:
                        longest_paths = [(i, j)]
                        max_length = path_length
                    elif path_length == max_length:
                        longest_paths.append((i, j))
                except nx.NetworkXNoPath:
                    continue
    
    if not longest_paths:
        return None
    
    # If multiple longest paths exist, use centrality to break ties
    if len(longest_paths) > 1:
        centrality = nx.betweenness_centrality(G)
        centrality_scores = [(pair, calculate_centrality_score(centrality, pair[0], pair[1])) 
                           for pair in longest_paths]
        return max(centrality_scores, key=lambda x: x[1])[0]
    
    return longest_paths[0]
def create_hybrid_edges(nodes, edges, percentage=0.20):
    """Create virtual edges using the hybrid approach with detailed timing."""
    print(f"\nStarting hybrid edge creation at {datetime.now()}")
    start_time = time.time()
    
    # Calculate limits
    total_allowed_edges = int(len(edges) * percentage)  # 20% of total edges
    strategy_limit = int(total_allowed_edges * 0.25)    # 25% of allowed edges for each strategy
    
    print(f"Total edge limit: {total_allowed_edges}")
    print(f"Per strategy limit: {strategy_limit}")
    
    # Keep track of total added edges
    total_edges_added = 0
    
    print(f"\nInitializing NetworkX graph...")
    G_hybrid = nx.Graph()
    G_hybrid.add_nodes_from(range(len(nodes)))
    G_hybrid.add_edges_from(edges.keys())
    
    # Initialize lists for each strategy's edges
    radial_edges = []
    cross_edges = []
    random_edges = []
    longest_path_edges = []
    
    # 1. Radial Connections
    print(f"\nStarting Radial Connections at {datetime.now()}")
    radial_start = time.time()
    
    main_angles = [0, 45, 90, 135]  # Main directions
    radial_edges_added = 0
    center = np.mean(nodes, axis=0)
    
    # Calculate angles once for all nodes
    node_angles = np.arctan2(nodes[:, 1] - center[1], nodes[:, 0] - center[0])
    node_angles = np.where(node_angles < 0, node_angles + 2*np.pi, node_angles)
    
    for angle in main_angles:
        angle_rad = np.radians(angle)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        projections = np.dot(nodes - center, direction)
        max_extent = np.max(projections) - np.min(projections)
        
        target_min = 0.45 * (max_extent/2)
        target_max = 0.75 * (max_extent/2)
        
        alignment_tolerance = 0.1
        cross_proj = np.abs(np.cross(np.stack([direction]*len(nodes)), nodes - center))
        aligned_mask = cross_proj < (alignment_tolerance * max_extent)
        
        pos_mask = (projections > target_min) & (projections < target_max) & aligned_mask
        pos_candidates = np.where(pos_mask)[0]
        
        neg_mask = (projections < -target_min) & (projections > -target_max) & aligned_mask
        neg_candidates = np.where(neg_mask)[0]
        
        if len(pos_candidates) > 0 and len(neg_candidates) > 0:
            optimal_dist = 0.6 * (max_extent/2)
            
            pos_dists = np.abs(projections[pos_candidates] - optimal_dist)
            best_pos = pos_candidates[np.argmin(pos_dists)]
            
            neg_dists = np.abs(projections[neg_candidates] + optimal_dist)
            best_neg = neg_candidates[np.argmin(neg_dists)]
            
            edge = tuple(sorted([best_pos, best_neg]))
            if edge not in edges and edge not in radial_edges:
                radial_edges.append(edge)
                G_hybrid.add_edge(*edge)
                radial_edges_added += 1
                total_edges_added += 1
    
    print(f"Radial Connections complete: {radial_edges_added} edges added")
    print(f"Time taken: {time.time() - radial_start:.2f} seconds")
    
    # 2. Distance-Constrained Random (25% of total limit)
    print(f"\nStarting Distance-Constrained Random at {datetime.now()}")
    random_start = time.time()
    random_edges_added = 0
    
    valid_pairs = []
    print("Calculating valid pairs...")
    
    for node1 in range(len(nodes)):
        if node1 % 50 == 0:
            print(f"Processing node {node1}/{len(nodes)}")
        for node2 in range(node1 + 1, len(nodes)):
            try:
                if nx.has_path(G_hybrid, node1, node2):
                    distance = nx.shortest_path_length(G_hybrid, node1, node2)
                    if distance >= 3:
                        valid_pairs.append((node1, node2))
            except nx.NetworkXError:
                continue
    
    print(f"Found {len(valid_pairs)} valid pairs")
    
    if valid_pairs:
        while random_edges_added < strategy_limit and valid_pairs:
            idx = np.random.randint(len(valid_pairs))
            node1, node2 = valid_pairs.pop(idx)
            edge = tuple(sorted([node1, node2]))
            if edge not in edges and edge not in random_edges:
                random_edges.append(edge)
                G_hybrid.add_edge(node1, node2)
                random_edges_added += 1
                total_edges_added += 1
    
    print(f"Distance-Constrained Random complete: {random_edges_added} edges added")
    print(f"Time taken: {time.time() - random_start:.2f} seconds")
    
    # 3. Iterative Longest Path (remaining capacity)
    print(f"\nStarting Iterative Longest Path at {datetime.now()}")
    longest_start = time.time()
    longest_edges_added = 0
    
    # Use remaining capacity
    while total_edges_added < total_allowed_edges:
        if longest_edges_added % 5 == 0:
            print(f"Processing longest path edge {longest_edges_added}")
        
        edge = find_longest_path_edge(G_hybrid, edges, nodes)
        if edge is None:
            break
            
        edge = tuple(sorted(edge))
        if edge not in edges and edge not in longest_path_edges:
            longest_path_edges.append(edge)
            G_hybrid.add_edge(*edge)
            longest_edges_added += 1
            total_edges_added += 1
    
    print(f"Iterative Longest Path complete: {longest_edges_added} edges added")
    print(f"Time taken: {time.time() - longest_start:.2f} seconds")
    
    # Final summary
    print(f"\nHybrid edge creation complete at {datetime.now()}")
    print(f"Radial edges added: {radial_edges_added}")
    print(f"Random edges added: {random_edges_added} (Limit: {strategy_limit})")
    print(f"Longest path edges added: {longest_edges_added}")
    print(f"Total edges added: {total_edges_added} (Limit: {total_allowed_edges})")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    # Combine all edges for final result
    all_edges_sequence = [
        radial_edges,
        random_edges,
        longest_path_edges
    ]

    return all_edges_sequence

def analyze_improvements(nodes, edges, hybrid_edges):
    """Analyze the improvement in graph connectivity."""
    G_original = nx.Graph(list(edges.keys()))
    G_hybrid = nx.Graph(list(edges.keys()) + hybrid_edges)
    
    # Calculate metrics
    avg_path_original = nx.average_shortest_path_length(G_original)
    avg_path_hybrid = nx.average_shortest_path_length(G_hybrid)
    
    diameter_original = nx.diameter(G_original)
    diameter_hybrid = nx.diameter(G_hybrid)
    
    # Calculate edge counts by region
    center = np.mean(nodes, axis=0)
    distances = np.linalg.norm(nodes - center, axis=1)
    distance_thirds = np.percentile(distances, [33, 66])
    
    def count_edges_by_region(edges_list):
        inner_count = 0
        middle_count = 0
        outer_count = 0
        cross_region_count = 0
        
        for edge in edges_list:
            n1, n2 = edge
            d1 = distances[n1]
            d2 = distances[n2]
            
            # Determine which regions the nodes belong to
            if d1 <= distance_thirds[0]:
                if d2 <= distance_thirds[0]:
                    inner_count += 1
                elif d2 <= distance_thirds[1]:
                    cross_region_count += 1
                else:
                    cross_region_count += 1
            elif d1 <= distance_thirds[1]:
                if d2 <= distance_thirds[0]:
                    cross_region_count += 1
                elif d2 <= distance_thirds[1]:
                    middle_count += 1
                else:
                    cross_region_count += 1
            else:
                if d2 <= distance_thirds[0]:
                    cross_region_count += 1
                elif d2 <= distance_thirds[1]:
                    cross_region_count += 1
                else:
                    outer_count += 1
                    
        return inner_count, middle_count, outer_count, cross_region_count
    
    orig_inner, orig_middle, orig_outer, orig_cross = count_edges_by_region(edges.keys())
    hybrid_inner, hybrid_middle, hybrid_outer, hybrid_cross = count_edges_by_region(hybrid_edges)
    
    print("\nGraph Analysis:")
    print(f"Original number of edges: {len(edges)}")
    print(f"Hybrid edges added: {len(hybrid_edges)}")
    
    print(f"\nEdge Distribution:")
    print("Original edges:")
    print(f"  Inner region: {orig_inner}")
    print(f"  Middle region: {orig_middle}")
    print(f"  Outer region: {orig_outer}")
    print(f"  Cross-region: {orig_cross}")
    
    print("\nHybrid edges added:")
    print(f"  Inner region: {hybrid_inner}")
    print(f"  Middle region: {hybrid_middle}")
    print(f"  Outer region: {hybrid_outer}")
    print(f"  Cross-region: {hybrid_cross}")
    
    print(f"\nConnectivity Metrics:")
    print(f"Average shortest path length:")
    print(f"  Original: {avg_path_original:.2f}")
    print(f"  With hybrid edges: {avg_path_hybrid:.2f}")
    print(f"  Improvement: {((avg_path_original - avg_path_hybrid) / avg_path_original * 100):.2f}%")
    
    print(f"\nGraph diameter:")
    print(f"  Original: {diameter_original}")
    print(f"  With hybrid edges: {diameter_hybrid}")
    print(f"  Improvement: {((diameter_original - diameter_hybrid) / diameter_original * 100):.2f}%")

def create_irregular_shapes():
    """Create different irregular mesh shapes for testing."""
    shapes = []
    
    # 1. L-shaped mesh
    def create_l_shape(n=15):
        points = []
        for i in range(n):
            for j in range(n):
                if i < n*2/3 or j < n/3:
                    x = (i/n)*2 - 1
                    y = (j/n)*2 - 1
                    points.append([x, y])
        return np.array(points)
    
    # 2. Circular mesh with hole
    def create_circular_mesh(n=15):
        points = []
        for i in range(n):
            for j in range(n):
                x = (i/n)*2 - 1
                y = (j/n)*2 - 1
                dist_from_center = np.sqrt(x**2 + y**2)
                # Exclude points in the center and outside the circle
                if 0.3 < dist_from_center < 1:
                    points.append([x, y])
        return np.array(points)
    
    # 3. Triangular mesh
    def create_triangular_mesh(n=15):
        points = []
        for i in range(n):
            for j in range(i+1):
                x = (i/n)*2 - 1
                y = ((j - i/2)/n)*2
                points.append([x, y])
        return np.array(points)
    
    # 4. S-shaped mesh
    def create_s_shape(n=15):
        points = []
        for i in range(n):
            for j in range(n):
                x = (i/n)*2 - 1
                y = (j/n)*2 - 1
                if (x < 0 and y > x + 0.5) or (x >= 0 and y < x - 0.5):
                    points.append([x, y])
        return np.array(points)
    
    def create_curved_shape(n=30):
        """Create a curved irregular shape (wave/spiral pattern)"""
        points = []
        t = np.linspace(0, 4*np.pi, n)
        
        # Create spiral-like pattern with varying radius
        for t_val in t:
            r = 0.3 + 0.1 * np.sin(2*t_val)  # Varying radius
            x = r * t_val/4 * np.cos(t_val)
            y = r * t_val/4 * np.sin(t_val)
            points.append([x, y])
        
        # Add some random perturbations for irregularity
        for i in range(n):
            angle = 2 * np.pi * i / n
            r = 0.3 + 0.1 * np.sin(3*angle)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            # Add some noise
            x += np.random.normal(0, 0.05)
            y += np.random.normal(0, 0.05)
            points.append([x, y])
        
        # Add some interior points
        for _ in range(n):
            r = np.random.uniform(0.1, 0.8)
            angle = np.random.uniform(0, 2*np.pi)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        
        # Scale to [-1, 1] range
        max_val = np.max(np.abs(points))
        points = points / max_val
        
        return points
    shapes.append(("Curved", create_curved_shape()))
    shapes.append(("L-Shape", create_l_shape()))
    shapes.append(("Circular with Hole", create_circular_mesh()))
    shapes.append(("Triangular", create_triangular_mesh()))
    shapes.append(("S-Shape", create_s_shape()))

    return shapes

def create_quad_mesh_from_points(points):
    """Create mesh edges from irregular point sets."""
    edges = {}
    kdtree = spatial.KDTree(points)
    
    for i in range(len(points)):
        # Find 4-8 nearest neighbors
        distances, indices = kdtree.query(points[i], k=8)
        for j in indices[1:5]:  # Use 4 closest neighbors
            if i < j:  # Avoid duplicate edges
                edges[(i, j)] = distances[indices == j][0]
    
    return edges

def visualize_all_shapes():
    shapes = create_irregular_shapes()
    
    for shape_name, nodes in shapes:
        print(f"\nProcessing {shape_name}")
        edges = create_quad_mesh_from_points(nodes)
        hybrid_edges_sequence = create_hybrid_edges(nodes, edges)
        
        # Visualization
        fig = plt.figure(figsize=(25, 15))
        fig.suptitle(f"Hybrid Strategy Results for {shape_name}", fontsize=16)
        
        strategies = [
            ("Original Mesh", 'gray', []),
            ("Radial Connections", 'blue', hybrid_edges_sequence[0]),
            ("Distance-Constrained\nRandom", 'magenta', hybrid_edges_sequence[1]),
            ("Iterative Longest Path", 'red', hybrid_edges_sequence[2]),
            ("Final Hybrid Result", 'gold', [edge for sublist in hybrid_edges_sequence for edge in sublist])
        ]
        analyze_improvements(nodes,edges,[edge for sublist in hybrid_edges_sequence for edge in sublist])
        for idx, (title, color, strategy_edges) in enumerate(strategies):
            
            ax = fig.add_subplot(2, 3, idx+1)
            
            # Plot original edges
            for edge in edges:
                n1, n2 = edge
                ax.plot([nodes[n1, 0], nodes[n2, 0]], 
                       [nodes[n1, 1], nodes[n2, 1]], 
                       'gray', alpha=0.5, linewidth=1, zorder=0)
            
            # Plot new edges for this strategy
            if strategy_edges:
                for edge in strategy_edges:
                    n1, n2 = edge
                    ax.plot([nodes[n1, 0], nodes[n2, 0]], 
                           [nodes[n1, 1], nodes[n2, 1]], 
                           color=color, linewidth=2, zorder=3)
            
            # Plot nodes and center
            center = np.mean(nodes, axis=0)
            ax.scatter(nodes[:, 0], nodes[:, 1], c='lightgray', s=50, zorder=1)
            ax.scatter(center[0], center[1], c='black', s=200, marker='*', zorder=2)
            
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.set_title(f"{title}\n({len(strategy_edges) if strategy_edges else 0} edges added)")
        
        plt.tight_layout()
        plt.show()

# Run visualization
visualize_all_shapes()