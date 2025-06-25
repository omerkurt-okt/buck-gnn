import os
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Subset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import chi2
from collections import Counter
import hashlib
import pandas as pd
from matplotlib.gridspec import GridSpec

# Set style for all plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_visualization_dirs(log_dir):
    """Create organized directories for different types of visualizations."""
    if log_dir is None:
        return None
    
    # Create main directories
    dirs = {
        'outliers': os.path.join(log_dir, 'outlier_analysis'),
        'distributions': os.path.join(log_dir, 'distributions'),
        'geometry': os.path.join(log_dir, 'geometry_analysis'),
        'splits': os.path.join(log_dir, 'split_analysis')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def detect_outliers(values, method='iqr', threshold=1.5):
    """
    Detect outliers using various methods with statistics.
    """
    if method == 'iqr':
        Q1 = np.percentile(values, 15)
        Q3 = np.percentile(values, 85)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (values >= lower_bound) & (values <= upper_bound)
        
        # Calculate additional statistics
        stats = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers_below': np.sum(values < lower_bound),
            'n_outliers_above': np.sum(values > upper_bound)
        }
        return mask, stats
    
    elif method == 'zscore':
        mean = np.mean(values)
        std = np.std(values)
        z_scores = np.abs((values - mean) / std)
        mask = z_scores < threshold
        
        stats = {
            'mean': mean,
            'std': std,
            'max_zscore': np.max(z_scores),
            'n_outliers': np.sum(~mask)
        }
        return mask, stats
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

def plot_outlier_analysis(values, mask, stats, title, save_path, use_log_scale=False):
    """Create outlier analysis visualization."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Distribution plot with outliers highlighted
    ax1 = fig.add_subplot(gs[0, :])
    if use_log_scale:
        values_plot = np.log10(values + 1e-10)
        xlabel = 'log10(Value)'
    else:
        values_plot = values
        xlabel = 'Value'
    
    sns.histplot(data=values_plot[mask], bins=50, color='blue', alpha=0.5, label='Normal', ax=ax1)
    sns.histplot(data=values_plot[~mask], bins=50, color='red', alpha=0.5, label='Outliers', ax=ax1)
    ax1.set_title(f'{title} Distribution')
    ax1.set_xlabel(xlabel)
    ax1.legend()
    
    # 2. Box plot
    ax2 = fig.add_subplot(gs[1, 0])
    sns.boxplot(y=values_plot, ax=ax2)
    ax2.set_title('Box Plot')
    ax2.set_ylabel(xlabel)
    
    # 3. Statistics table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    stats_text = [
        f"Total samples: {len(values)}",
        f"Outliers: {np.sum(~mask)} ({100*np.sum(~mask)/len(values):.2f}%)",
        f"Mean: {np.mean(values):.2e}",
        f"Std: {np.std(values):.2e}",
        f"Min: {np.min(values):.2e}",
        f"Max: {np.max(values):.2e}"
    ]
    
    if 'Q1' in stats:
        stats_text.extend([
            f"Q1: {stats['Q1']:.2e}",
            f"Q3: {stats['Q3']:.2e}",
            f"IQR: {stats['IQR']:.2e}",
            f"Lower bound: {stats['lower_bound']:.2e}",
            f"Upper bound: {stats['upper_bound']:.2e}",
            f"Outliers below: {stats['n_outliers_below']}",
            f"Outliers above: {stats['n_outliers_above']}"
        ])
    
    ax3.text(0.1, 0.9, '\n'.join(stats_text), va='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_2d_outlier_analysis(x, y, mask, title, save_path, use_log_scale=False):
    """Create comprehensive 2D outlier analysis visualization."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Scatter plot
    ax1 = fig.add_subplot(gs[0, :])
    if use_log_scale:
        x_plot = np.log10(x + 1e-10)
        y_plot = np.log10(y + 1e-10)
        xlabel = 'log10(X)'
        ylabel = 'log10(Y)'
    else:
        x_plot = x
        y_plot = y
        xlabel = 'X'
        ylabel = 'Y'
    
    ax1.scatter(x_plot[mask], y_plot[mask], c='blue', label='Normal', alpha=0.5, s=20)
    ax1.scatter(x_plot[~mask], y_plot[~mask], c='red', label='Outliers', alpha=0.5, s=20)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()
    
    # 2. Density plot
    ax2 = fig.add_subplot(gs[1, 0])
    sns.kdeplot(data=pd.DataFrame({'x': x_plot[mask], 'y': y_plot[mask]}),
                x='x', y='y', ax=ax2, cmap='Blues')
    ax2.set_title('Density Plot (Normal Points)')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    
    # 3. Statistics
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    stats_text = [
        f"Total points: {len(x)}",
        f"Normal points: {np.sum(mask)}",
        f"Outliers: {np.sum(~mask)} ({100*np.sum(~mask)/len(x):.2f}%)",
        f"\nX statistics:",
        f"Mean: {np.mean(x):.2e}",
        f"Std: {np.std(x):.2e}",
        f"Min: {np.min(x):.2e}",
        f"Max: {np.max(x):.2e}",
        f"\nY statistics:",
        f"Mean: {np.mean(y):.2e}",
        f"Std: {np.std(y):.2e}",
        f"Min: {np.min(y):.2e}",
        f"Max: {np.max(y):.2e}"
    ]
    
    ax3.text(0.1, 0.9, '\n'.join(stats_text), va='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def identify_geometry_groups(dataset):
    """
    Identify groups of graphs that share the same geometry using a robust hashing approach.
    Returns a dictionary mapping geometry hash to list of indices.
    """
    geometry_groups = {}
    
    # Store verification data for visualization
    verification_data = {
        'coord_diffs': [],
        'edge_diffs': [],
        'group_sizes': []
    }
    
    for idx, data in enumerate(dataset):
        # Extract node coordinates and edge information
        node_coords = data.x.numpy()[:, :2]  # Use only x,y coordinates
        edge_index = data.edge_index.numpy()
        
        # Filter out virtual edges (only keep edges where the last feature is 0)
        real_edge_mask = data.edge_attr.numpy()[:, -1] == 0
        real_edge_indices = edge_index[:, real_edge_mask]
        
        # Round coordinates to reduce floating point precision issues
        nodes_rounded = np.round(node_coords, decimals=3)
        
        # Get bounding box and dimensions
        min_coords = np.min(nodes_rounded, axis=0)
        max_coords = np.max(nodes_rounded, axis=0)
        dimensions = max_coords - min_coords
        
        # Normalize coordinates to make geometry identification more robust
        nodes_normalized = (nodes_rounded - min_coords) / (dimensions + 1e-8)
        
        # Create sorted edge list for consistent ordering
        edge_list = sorted([tuple(sorted([int(real_edge_indices[0, i]), 
                                        int(real_edge_indices[1, i])])) 
                          for i in range(real_edge_indices.shape[1])])
        
        # Create node connectivity pattern
        node_connections = {}
        for edge in edge_list:
            n1, n2 = edge
            if n1 not in node_connections:
                node_connections[n1] = []
            if n2 not in node_connections:
                node_connections[n2] = []
            node_connections[n1].append(n2)
            node_connections[n2].append(n1)
        
        # Sort connections for each node
        for node in node_connections:
            node_connections[node] = sorted(node_connections[node])
        
        # Create shape info string
        shape_info = [
            f"{len(nodes_rounded)}_{len(edge_list)}",  # Basic size info
            f"{dimensions[0]:.3f}_{dimensions[1]:.3f}",  # Dimensions
        ]
        
        # Add normalized coordinate string
        coord_str = "_".join(f"{x:.3f}_{y:.3f}" for x, y in nodes_normalized)
        shape_info.append(coord_str)
        
        # Add connectivity pattern
        for node in sorted(node_connections.keys()):
            if node_connections[node]:
                conn_str = f"{node}:" + ",".join(map(str, node_connections[node]))
                shape_info.append(conn_str)
        
        # Create hash
        hasher = hashlib.sha256("_".join(shape_info).encode())
        geom_hash = hasher.hexdigest()
        
        if geom_hash not in geometry_groups:
            geometry_groups[geom_hash] = []
        geometry_groups[geom_hash].append(idx)
        
        # Collect verification data
        if len(geometry_groups[geom_hash]) > 1:
            ref_idx = geometry_groups[geom_hash][0]
            ref_coords = dataset[ref_idx].x.numpy()[:, :2]
            ref_edges = dataset[ref_idx].edge_index.numpy()[:, 
                        dataset[ref_idx].edge_attr.numpy()[:, -1] == 0]
            
            # Calculate differences
            coord_diff = np.max(np.abs(ref_coords - node_coords))
            verification_data['coord_diffs'].append(coord_diff)
            
            ref_edges_set = {tuple(sorted(e)) for e in ref_edges.T}
            curr_edges_set = {tuple(sorted(e)) for e in real_edge_indices.T}
            edge_diff = len(ref_edges_set.symmetric_difference(curr_edges_set))
            verification_data['edge_diffs'].append(edge_diff)
    
    verification_data['group_sizes'] = [len(indices) for indices in geometry_groups.values()]
    
    # Print statistics
    print("\nGeometry Group Statistics:")
    print(f"Total number of unique geometries: {len(geometry_groups)}")
    group_sizes = verification_data['group_sizes']
    print(f"Average cases per geometry: {np.mean(group_sizes):.2f}")
    print(f"Min cases per geometry: {np.min(group_sizes)}")
    print(f"Max cases per geometry: {np.max(group_sizes)}")
    
    # Print histogram of group sizes
    size_counts = Counter(group_sizes)
    print("\nGeometry group size distribution:")
    for size, count in sorted(size_counts.items()):
        print(f"Groups with {size} cases: {count}")
    
    # Print largest groups
    largest_groups = sorted(geometry_groups.items(), key=lambda x: len(x[1]), reverse=True)
    print("\nLargest geometry groups:")
    for i, (hash_val, indices) in enumerate(largest_groups[:5]):
        print(f"\nGroup {i+1}:")
        print(f"Hash: {hash_val[:8]}...")
        print(f"Size: {len(indices)} cases")
        print(f"Sample indices: {indices[:5]}")
    
    return geometry_groups, verification_data

def plot_geometry_verification(verification_data, save_dir):
    """Create comprehensive geometry verification plots."""
    # 1. Group Size Distribution
    plt.figure(figsize=(12, 6))
    group_sizes = verification_data['group_sizes']
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=group_sizes, bins=30)
    plt.title('Distribution of Group Sizes')
    plt.xlabel('Number of Cases per Group')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    size_counts = Counter(group_sizes)
    sizes = sorted(size_counts.keys())
    counts = [size_counts[s] for s in sizes]
    plt.bar(sizes, counts)
    plt.title('Group Size Histogram')
    plt.xlabel('Group Size')
    plt.ylabel('Number of Groups')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geometry_group_sizes.png'))
    plt.close()
    
    # 2. Verification Metrics
    if verification_data['coord_diffs'] and verification_data['edge_diffs']:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(data=verification_data['coord_diffs'], bins=30)
        plt.title('Distribution of Coordinate Differences')
        plt.xlabel('Max Coordinate Difference')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=verification_data['edge_diffs'], bins=30)
        plt.title('Distribution of Edge Differences')
        plt.xlabel('Number of Different Edges')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'geometry_verification_metrics.png'))
        plt.close()
        
        # 3. Correlation Plot
        if len(verification_data['coord_diffs']) > 1:
            plt.figure(figsize=(8, 8))
            plt.scatter(verification_data['coord_diffs'], 
                       verification_data['edge_diffs'], 
                       alpha=0.5)
            plt.title('Correlation between Coordinate and Edge Differences')
            plt.xlabel('Max Coordinate Difference')
            plt.ylabel('Number of Different Edges')
            plt.savefig(os.path.join(save_dir, 'geometry_difference_correlation.png'))
            plt.close()
def create_bins(dataset, prediction_type, n_bins=10):
    """Create bins based on prediction type with enhanced binning strategy."""
    if prediction_type == "static":
        # Calculate maximum displacement and stress for each graph
        disp_magnitudes = []
        von_mises_max = []
        
        for data in dataset:
            values = data.y.numpy()
            displacements = values[:, :2]
            stresses = values[:, 2:]
            
            # Maximum displacement magnitude
            disp_mag = np.sqrt(np.sum(displacements**2, axis=1))
            disp_magnitudes.append(np.max(disp_mag))
            
            # Maximum von Mises stress
            von_mises = np.sqrt(stresses[:, 0]**2 - stresses[:, 0]*stresses[:, 1] + 
                              stresses[:, 1]**2 + 3*stresses[:, 2]**2)
            von_mises_max.append(np.max(von_mises))
        
        disp_magnitudes = np.array(disp_magnitudes)
        von_mises_max = np.array(von_mises_max)
        
        # Create bins
        _, disp_edges = np.histogram(disp_magnitudes, bins='auto')
        _, stress_edges = np.histogram(von_mises_max, bins='auto')

        disp_bins = np.digitize(disp_magnitudes, disp_edges) - 1
        stress_bins = np.digitize(von_mises_max, stress_edges) - 1

        n_disp_bins = len(np.unique(disp_bins))
        combined_bins = disp_bins * n_disp_bins + stress_bins
        values = (disp_magnitudes, von_mises_max)
        
        bin_info = {
            'disp_edges': disp_edges,
            'stress_edges': stress_edges,
            'disp_bins': disp_bins,
            'stress_bins': stress_bins
        }
        
    elif prediction_type == "buckling":
        values = np.array([data.y.item() for data in dataset])
        hist, edges = np.histogram(values, bins='auto')
        combined_bins = np.digitize(values, edges) - 1
        
        # Analyze dataset distribution
        min_val = np.min(values)
        max_val = np.max(values)
        
        print("\nAnalyzing eigenvalue distribution:")
        print(f"Range: [{min_val:.5e}, {max_val:.5e}]")
        
        # Calculate density for each region
        total_samples = len(values)
        cumulative_samples = 0
        
        # Analyze each region in the histogram
        for i, count in enumerate(hist):
            region_start = edges[i]
            region_end = edges[i+1]
            region_width = region_end - region_start
            density = count / region_width
            cumulative_samples += count
            
            print(f"Region [{region_start:.3e}, {region_end:.3e}]: "
                f"{count} samples ({count/total_samples*100:.1f}%), "
                f"density: {density:.1f} samples/unit")
            
        bin_info = {
            'edges': edges,
            'values': values
        }
        

    elif prediction_type == "modeshape":
        # Calculate maximum magnitude and pattern features
        magnitudes = []
        normalized_shapes = []
        
        for data in dataset:
            mode_shape = data.y.numpy()
            shape_magnitudes = np.sqrt(np.sum(mode_shape**2, axis=1))
            max_magnitude = np.max(shape_magnitudes)
            magnitudes.append(max_magnitude)
            
            # Normalize shape for pattern analysis
            normalized_shape = mode_shape / (max_magnitude + 1e-8)
            normalized_shapes.append(normalized_shape.flatten())
        
        magnitudes = np.array(magnitudes)
        normalized_shapes = np.array(normalized_shapes)
        
        # Create magnitude bins using log-space
        _, magnitude_edges = np.histogram(magnitudes, bins='auto')
        magnitude_bins = np.digitize(magnitudes, magnitude_edges) - 1
        
        # Create pattern bins using PCA and clustering
        pca = PCA(n_components=min(5, normalized_shapes.shape[1]))
        pattern_features = pca.fit_transform(normalized_shapes)
        
        # Use KMeans with multiple initializations
        kmeans = KMeans(n_clusters=len(np.unique(magnitude_bins)), n_init=10, random_state=42)
        pattern_clusters = kmeans.fit_predict(pattern_features)
        
        # Combine bins
        n_mag_bins = len(np.unique(magnitude_bins))
        combined_bins = magnitude_bins * n_mag_bins + pattern_clusters
        values = (magnitudes, pattern_features)
        bin_info = {
            'magnitude_edges': magnitude_edges,
            'magnitude_bins': magnitude_bins,
            'pattern_clusters': pattern_clusters,
            'pca': pca,
            'kmeans': kmeans
        }
    
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    return combined_bins, values, bin_info

def plot_binning_analysis(values, bin_info, prediction_type, save_dir):
    """Create comprehensive binning analysis visualization."""
    if prediction_type == "static":
        disp_magnitudes, von_mises_max = values
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)
        
        # Displacement distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data=disp_magnitudes, bins=30, ax=ax1)
        ax1.set_title('Displacement Distribution')
        ax1.set_xlabel('Displacement')
        
        # Stress distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(data=von_mises_max, bins=30, ax=ax2)
        ax2.set_title('Von Mises Stress Distribution')
        ax2.set_xlabel('Stress')
        
        # 2D histogram
        ax3 = fig.add_subplot(gs[1, :])
        plt.hist2d(disp_magnitudes, von_mises_max, bins=30)
        plt.colorbar(label='Count')
        ax3.set_title('2D Distribution')
        ax3.set_xlabel('Displacement')
        ax3.set_ylabel('Stress')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'static_binning_analysis.png'))
        plt.close()
        
    if prediction_type == "buckling":
        fig = plt.figure(figsize=(15, 15))
        
        # Original distribution with density analysis
        plt.subplot(3, 1, 1)
        plt.hist(values, bins='auto', alpha=0.7, color='skyblue')
        plt.title('Original Distribution with Auto Bins')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Count')
        
        # Distribution with adaptive bins
        plt.subplot(3, 1, 2)
        counts, edges, patches = plt.hist(values, bins=bin_info['edges'], 
                                        alpha=0.7, color='lightgreen')
        plt.title('Distribution with Adaptive Bins')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Count')
        
        # Add count labels on top of bars
        for i in range(len(counts)):
            if counts[i] > 0:
                plt.text(edges[i], counts[i], f'{int(counts[i])}',
                        ha='center', va='bottom')
        
        # Log-scale view
        plt.subplot(3, 1, 3)
        plt.hist(values, bins=bin_info['edges'], alpha=0.7, color='lightgreen')
        plt.xscale('log')
        plt.title('Distribution with Adaptive Bins (Log Scale)')
        plt.xlabel('Eigenvalue (log scale)')
        plt.ylabel('Count')
        
        # Add statistics
        stats = f"Total samples: {len(values)}\n"
        stats += f"Mean: {np.mean(values):.3e}\n"
        stats += f"Median: {np.median(values):.3e}\n"
        stats += f"Std: {np.std(values):.3e}\n"
        stats += f"Min: {np.min(values):.3e}\n"
        stats += f"Max: {np.max(values):.3e}\n"
        stats += f"Number of bins: {len(bin_info['edges'])-1}"
        
        plt.figtext(0.95, 0.95, stats,
                   bbox=dict(facecolor='white', alpha=0.8),
                   verticalalignment='top',
                   horizontalalignment='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'buckling_binning_analysis.png'))
        plt.close()
        
    else:  # modeshape
        magnitudes, pattern_features = values
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)
        
        # Magnitude distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data=np.log10(magnitudes), bins=30, ax=ax1)
        ax1.set_title('Log10 Magnitude Distribution')
        ax1.set_xlabel('Log10(Magnitude)')
        
        # Pattern distribution
        ax2 = fig.add_subplot(gs[0, 1])
        plt.scatter(pattern_features[:, 0], pattern_features[:, 1], 
                   c=bin_info['pattern_clusters'], cmap='tab10', alpha=0.5)
        ax2.set_title('Pattern Clusters')
        ax2.set_xlabel('PCA Component 1')
        ax2.set_ylabel('PCA Component 2')
        
        # PCA explained variance
        ax3 = fig.add_subplot(gs[1, :])
        explained_var = np.cumsum(bin_info['pca'].explained_variance_ratio_)
        plt.plot(range(1, len(explained_var) + 1), explained_var, '-o')
        plt.title('PCA Explained Variance Ratio')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'modeshape_binning_analysis.png'))
        plt.close()
def detect_buckling_outliers(dataset, log_dir=None):
    """Detect outliers for buckling analysis using simple statistical approach."""
    eigenvalues = np.array([data.y.item() for data in dataset])
    
    # Calculate statistics
    Q1 = np.percentile(eigenvalues, 15)
    Q3 = np.percentile(eigenvalues, 85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    
    # Create mask for non-outliers
    mask = (eigenvalues >= lower_bound) & (eigenvalues <= upper_bound)
    
    print("\nBuckling Analysis Statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Min eigenvalue: {np.min(eigenvalues):.6f}")
    print(f"Max eigenvalue: {np.max(eigenvalues):.6f}")
    print(f"Mean eigenvalue: {np.mean(eigenvalues):.6f}")
    print(f"Median eigenvalue: {np.median(eigenvalues):.6f}")
    print(f"\nOutlier Detection:")
    print(f"Q1: {Q1:.6f}")
    print(f"Q3: {Q3:.6f}")
    print(f"IQR: {IQR:.6f}")
    print(f"Lower bound: {lower_bound:.6f}")
    print(f"Upper bound: {upper_bound:.6f}")
    print(f"Samples after outlier removal: {np.sum(mask)}")
    print(f"Outliers removed: {len(dataset) - np.sum(mask)}")

    return mask

def detect_static_outliers(dataset, log_dir=None):
    """Detect outliers for static analysis."""
    all_values = np.vstack([data.y.numpy() for data in dataset])
    displacements = all_values[:, :2]
    stresses = all_values[:, 2:]
    
    # Calculate magnitudes per graph
    disp_magnitudes = np.array([np.max(np.sqrt(np.sum(data.y.numpy()[:, :2]**2, axis=1))) 
                               for data in dataset])
    
    von_mises = np.array([np.max(np.sqrt(stresses[:, 0]**2 - stresses[:, 0]*stresses[:, 1] + 
                                        stresses[:, 1]**2 + 3*stresses[:, 2]**2))
                          for data in dataset])
    
    # Remove any invalid values
    valid_mask = np.isfinite(disp_magnitudes) & np.isfinite(von_mises)
    disp_magnitudes = disp_magnitudes[valid_mask]
    von_mises = von_mises[valid_mask]
    
    # Detect outliers
    disp_mask, disp_stats = detect_outliers(disp_magnitudes)
    stress_mask, stress_stats = detect_outliers(von_mises)
    combined_mask = disp_mask & stress_mask
    
    # Create final mask including invalid values
    final_mask = np.zeros(len(dataset), dtype=bool)
    final_mask[valid_mask] = combined_mask
    
    print("\nStatic Analysis Outlier Detection:")
    print(f"Total samples: {len(dataset)}")
    print(f"Invalid values removed: {np.sum(~valid_mask)}")
    print(f"Samples after outlier removal: {np.sum(final_mask)}")
    print(f"Outliers removed: {len(dataset) - np.sum(final_mask)}")
    
    if log_dir:
        vis_dir = os.path.join(log_dir, 'outlier_analysis')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Individual analysis plots
        plot_outlier_analysis(
            disp_magnitudes, disp_mask,
            disp_stats,
            'Maximum Displacement Distribution',
            os.path.join(vis_dir, 'displacement_outliers.png')
        )
        
        plot_outlier_analysis(
            von_mises, stress_mask,
            stress_stats,
            'Maximum Von Mises Stress Distribution',
            os.path.join(vis_dir, 'stress_outliers.png')
        )
        
        # 2. Combined analysis
        plot_2d_outlier_analysis(
            disp_magnitudes, von_mises,
            combined_mask,
            'Displacement vs Stress Distribution',
            os.path.join(vis_dir, 'static_combined_outliers.png')
        )
        
        # 3. Correlation analysis
        plt.figure(figsize=(10, 10))
        plt.scatter(disp_magnitudes[combined_mask],
                   von_mises[combined_mask],
                   c='blue', label='Normal', alpha=0.5, s=20)
        plt.scatter(disp_magnitudes[~combined_mask],
                   von_mises[~combined_mask],
                   c='red', label='Outliers', alpha=0.5, s=20)
        
        plt.title('Displacement-Stress Correlation')
        plt.xlabel('Maximum Displacement')
        plt.ylabel('Maximum Von Mises Stress')
        plt.legend()
        
        # Add correlation coefficient
        corr = np.corrcoef(disp_magnitudes, von_mises)[0, 1]
        plt.text(0.02, 0.98, f'Correlation: {corr:.3f}',
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(vis_dir, 'static_correlation.png'))
        plt.close()
    
    return final_mask
def detect_modeshape_outliers(dataset, log_dir=None):
    """Detect outliers for mode shape analysis with enhanced visualization."""
    magnitudes = []
    normalized_shapes = []
    valid_indices = []
    
    for idx, data in enumerate(dataset):
        mode_shape = data.y.numpy()
        
        # Calculate magnitude
        shape_magnitudes = np.sqrt(np.sum(mode_shape**2, axis=1))
        max_magnitude = np.max(shape_magnitudes)
        
        if np.isfinite(max_magnitude):  # Only include valid values
            magnitudes.append(max_magnitude)
            # Normalize shape for pattern analysis
            normalized_shape = mode_shape / (max_magnitude + 1e-8)
            normalized_shapes.append(normalized_shape.flatten())
            valid_indices.append(idx)
    
    magnitudes = np.array(magnitudes)
    normalized_shapes = np.array(normalized_shapes)
    
    # Detect magnitude outliers
    magnitude_mask, magnitude_stats = detect_outliers(np.log10(magnitudes + 1e-10))
    
    # Detect pattern outliers using PCA and Mahalanobis distance
    pca = PCA(n_components=min(5, normalized_shapes.shape[1]))
    pattern_features = pca.fit_transform(normalized_shapes)
    
    pattern_mean = np.mean(pattern_features, axis=0)
    pattern_cov = np.cov(pattern_features.T)
    
    def mahalanobis_distance(x, mean, cov):
        diff = x - mean
        return np.sqrt(diff.dot(np.linalg.inv(cov)).dot(diff.T))
    
    m_distances = np.array([mahalanobis_distance(x, pattern_mean, pattern_cov) 
                           for x in pattern_features])
    
    chi2_threshold = chi2.ppf(0.99, df=pattern_features.shape[1])
    pattern_mask = m_distances < chi2_threshold
    
    # Combine masks
    combined_mask = magnitude_mask & pattern_mask
    
    # Create final mask for all datasets
    final_mask = np.zeros(len(dataset), dtype=bool)
    final_mask[valid_indices] = combined_mask
    
    print("\nMode Shape Analysis Outlier Detection:")
    print(f"Total samples: {len(dataset)}")
    print(f"Invalid values removed: {len(dataset) - len(valid_indices)}")
    print(f"Samples after outlier removal: {np.sum(final_mask)}")
    print(f"Outliers removed: {len(dataset) - np.sum(final_mask)}")
    
    if log_dir:
        vis_dir = os.path.join(log_dir, 'outlier_analysis')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Magnitude analysis
        plot_outlier_analysis(
            magnitudes, magnitude_mask,
            magnitude_stats,
            'Mode Shape Magnitude Distribution',
            os.path.join(vis_dir, 'modeshape_magnitude_outliers.png'),
            use_log_scale=True
        )
        
        # 2. Pattern analysis
        plt.figure(figsize=(15, 10))
        
        # PCA components scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(pattern_features[pattern_mask, 0],
                   pattern_features[pattern_mask, 1],
                   c='blue', label='Normal', alpha=0.5, s=20)
        plt.scatter(pattern_features[~pattern_mask, 0],
                   pattern_features[~pattern_mask, 1],
                   c='red', label='Outliers', alpha=0.5, s=20)
        plt.title('First Two PCA Components')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        
        # Mahalanobis distance distribution
        plt.subplot(2, 2, 2)
        plt.hist(m_distances[pattern_mask], bins=30, alpha=0.5,
                label='Normal', color='blue')
        plt.hist(m_distances[~pattern_mask], bins=30, alpha=0.5,
                label='Outliers', color='red')
        plt.axvline(chi2_threshold, color='k', linestyle='--',
                   label=f'Threshold ({chi2_threshold:.2f})')
        plt.title('Mahalanobis Distance Distribution')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Explained variance ratio
        plt.subplot(2, 2, 3)
        explained_var = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(explained_var) + 1), explained_var, '-o')
        plt.title('PCA Explained Variance Ratio')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        
        # Combined analysis
        plt.subplot(2, 2, 4)
        plt.scatter(np.log10(magnitudes), m_distances,
                   c=['blue' if m else 'red' for m in combined_mask],
                   alpha=0.5, s=20)
        plt.axhline(chi2_threshold, color='k', linestyle='--',
                   label='Pattern Threshold')
        plt.title('Magnitude vs Pattern Analysis')
        plt.xlabel('Log10(Magnitude)')
        plt.ylabel('Mahalanobis Distance')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'modeshape_pattern_analysis.png'))
        plt.close()
        
        # 3. Mode shape visualization
        if len(normalized_shapes) > 0:
            # Visualize representative mode shapes
            plt.figure(figsize=(15, 5))
            
            # Normal example
            normal_idx = np.where(combined_mask)[0][0]
            plt.subplot(1, 3, 1)
            plt.imshow(normalized_shapes[normal_idx].reshape(-1, 3))
            plt.title('Normal Mode Shape Example')
            plt.colorbar()
            
            # Outlier example (if any)
            if np.sum(~combined_mask) > 0:
                outlier_idx = np.where(~combined_mask)[0][0]
                plt.subplot(1, 3, 2)
                plt.imshow(normalized_shapes[outlier_idx].reshape(-1, 3))
                plt.title('Outlier Mode Shape Example')
                plt.colorbar()
            
            # Difference plot (if outlier exists)
            if np.sum(~combined_mask) > 0:
                plt.subplot(1, 3, 3)
                diff = (normalized_shapes[normal_idx] - 
                       normalized_shapes[outlier_idx]).reshape(-1, 3)
                plt.imshow(diff)
                plt.title('Difference Plot')
                plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'modeshape_examples.png'))
            plt.close()
    
    return final_mask

def split_geometry_group(indices, all_bins, lengths, is_abundant=False):
    """
    Split a group of indices with priority for training set.
    
    Args:
        indices: List of indices to split
        all_bins: Bin assignments for all samples
        lengths: Target split ratios
        is_abundant: Whether this geometry/bin combination is abundant
    """
    if len(indices) == 0:
        return [[] for _ in lengths]
    
    # Group indices by bin
    bin_groups = {}
    for idx in indices:
        bin_val = all_bins[idx]
        if bin_val not in bin_groups:
            bin_groups[bin_val] = []
        bin_groups[bin_val].append(idx)
    
    # Calculate target sizes
    total_samples = len(indices)
    target_sizes = [int(total_samples * length) for length in lengths]
    target_sizes[-1] = total_samples - sum(target_sizes[:-1])
    
    # Initialize splits
    split_indices = [[] for _ in lengths]
    split_sizes = [0] * len(lengths)
    
    if not is_abundant:
        # First ensure training set gets at least one sample from each bin
        for bin_indices in bin_groups.values():
            if len(bin_indices) > 0:
                np.random.shuffle(bin_indices)
                split_indices[0].append(bin_indices[0])
                split_sizes[0] += 1
                bin_indices = bin_indices[1:]
    
    # Distribute remaining samples
    for bin_indices in bin_groups.values():
        np.random.shuffle(bin_indices)
        for i, idx in enumerate(bin_indices):
            split_idx = i % len(lengths)
            if split_sizes[split_idx] < target_sizes[split_idx]:
                split_indices[split_idx].append(idx)
                split_sizes[split_idx] += 1
            else:
                # Find next available split
                for j in range(len(lengths)):
                    if split_sizes[j] < target_sizes[j]:
                        split_indices[j].append(idx)
                        split_sizes[j] += 1
                        break
    
    return split_indices


def plot_split_analysis(splits, dataset, prediction_type, log_dir):
    """Create comprehensive visualization of dataset splits."""
    vis_dir = os.path.join(log_dir, 'split_analysis')
    os.makedirs(vis_dir, exist_ok=True)
    
    if prediction_type == "buckling":
        values = [data.y.item() for data in dataset]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, (name, split) in enumerate(zip(['Train', 'Val', 'Test'], splits)):
            split_values = [values[idx] for idx in split.indices]
            
            # Original scale
            sns.histplot(data=split_values, bins=30, ax=axes[0, i])
            axes[0, i].set_title(f'{name} Set Distribution')
            axes[0, i].set_xlabel('Eigenvalue')
            
            # Add statistics
            stats = f'n={len(split_values)}\n'
            stats += f'mean={np.mean(split_values):.2e}\n'
            stats += f'std={np.std(split_values):.2e}'
            axes[0, i].text(0.95, 0.95, stats,
                          transform=axes[0, i].transAxes,
                          verticalalignment='top',
                          horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Cumulative distribution
            values_sorted = np.sort(split_values)
            cumulative = np.arange(1, len(values_sorted) + 1) / len(values_sorted)
            axes[1, i].plot(values_sorted, cumulative)
            axes[1, i].set_title(f'{name} Set Cumulative Distribution')
            axes[1, i].set_xlabel('Eigenvalue')
            axes[1, i].set_ylabel('Cumulative Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'buckling_split_distribution.png'))
        plt.close()
        
    elif prediction_type == "static":
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        for i, (name, split) in enumerate(zip(['Train', 'Val', 'Test'], splits)):
            split_values = [data.y.numpy() for idx, data in enumerate(dataset) 
                          if idx in split.indices]
            split_values = np.vstack(split_values)
            
            # Displacement magnitude
            disp_mag = np.sqrt(np.sum(split_values[:, :2]**2, axis=1))
            sns.histplot(data=disp_mag, bins=30, ax=axes[0, i])
            axes[0, i].set_title(f'{name} Set Displacement')
            axes[0, i].set_xlabel('Displacement Magnitude')
            
            # Von Mises stress
            stresses = split_values[:, 2:]
            von_mises = np.sqrt(stresses[:, 0]**2 - stresses[:, 0]*stresses[:, 1] + 
                              stresses[:, 1]**2 + 3*stresses[:, 2]**2)
            sns.histplot(data=von_mises, bins=30, ax=axes[1, i])
            axes[1, i].set_title(f'{name} Set Stress')
            axes[1, i].set_xlabel('Von Mises Stress')
            
            # 2D scatter plot
            axes[2, i].scatter(disp_mag, von_mises, alpha=0.5, s=20)
            axes[2, i].set_title(f'{name} Set Correlation')
            axes[2, i].set_xlabel('Displacement Magnitude')
            axes[2, i].set_ylabel('Von Mises Stress')
            
            # Add statistics
            stats = f'n={len(split_values)}\n'
            stats += f'max_disp={np.max(disp_mag):.2e}\n'
            stats += f'max_stress={np.max(von_mises):.2e}\n'
            stats += f'corr={np.corrcoef(disp_mag, von_mises)[0,1]:.2f}'
            axes[0, i].text(0.95, 0.95, stats,
                          transform=axes[0, i].transAxes,
                          verticalalignment='top',
                          horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'static_split_distribution.png'))
        plt.close()
        
        
    else:  # modeshape
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, (name, split) in enumerate(zip(['Train', 'Val', 'Test'], splits)):
            split_values = [data.y.numpy() for idx, data in enumerate(dataset) 
                          if idx in split.indices]
            
            # Calculate magnitudes
            magnitudes = [np.max(np.sqrt(np.sum(v**2, axis=1))) for v in split_values]
            
            # Magnitude distribution
            sns.histplot(data=np.log10(magnitudes), bins=30, ax=axes[0, i])
            axes[0, i].set_title(f'{name} Set Magnitude')
            axes[0, i].set_xlabel('Log10(Magnitude)')
            
            # Pattern analysis using PCA
            normalized_shapes = [v / (np.max(np.sqrt(np.sum(v**2, axis=1))) + 1e-8)
                               for v in split_values]
            normalized_shapes = np.vstack([s.flatten() for s in normalized_shapes])
            
            if len(normalized_shapes) > 1:
                pca = PCA(n_components=2)
                pattern_features = pca.fit_transform(normalized_shapes)
                
                axes[1, i].scatter(pattern_features[:, 0], pattern_features[:, 1],
                                 alpha=0.5, s=20)
                axes[1, i].set_title(f'{name} Set Pattern')
                axes[1, i].set_xlabel('PC1')
                axes[1, i].set_ylabel('PC2')
            
            # Add statistics
            stats = f'n={len(split_values)}\n'
            stats += f'mean_mag={np.mean(magnitudes):.2e}\n'
            stats += f'std_mag={np.std(magnitudes):.2e}'
            axes[0, i].text(0.95, 0.95, stats,
                          transform=axes[0, i].transAxes,
                          verticalalignment='top',
                          horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'modeshape_split_distribution.png'))
        plt.close()

def dataset_split(dataset, prediction_type="static", lengths=[0.85, 0.15],
                 remove_outliers=True, n_bins=10, log_dir=None):
    """
    Split dataset while maintaining distribution and target ratios.
    Enhanced version with better visualization and analysis.
    """
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        vis_dirs = create_visualization_dirs(log_dir)
    
    # # Remove outliers if requested
    # if remove_outliers:
    #     if prediction_type == "buckling":
    #         mask = detect_buckling_outliers(dataset, log_dir)
    #     elif prediction_type == "static":
    #         mask = detect_static_outliers(dataset, log_dir)
    #     elif prediction_type == "modeshape":
    #         mask = detect_modeshape_outliers(dataset, log_dir)
    #     else:
    #         raise ValueError(f"Unknown prediction type: {prediction_type}")
        
    #     dataset = [data for i, data in enumerate(dataset) if mask[i]]
    
    # Create bins for response values
    bins, values, bin_info = create_bins(dataset, prediction_type, n_bins)
    
    if log_dir:
        plot_binning_analysis(values, bin_info, prediction_type, vis_dirs['distributions'])
    
    # Identify geometry groups
    geometry_groups, verification_data = identify_geometry_groups(dataset)
    
    if log_dir:
        plot_geometry_verification(verification_data, vis_dirs['geometry'])
    
    # Calculate target sizes
    total_samples = len(dataset)
    target_sizes = [int(total_samples * length) for length in lengths]
    target_sizes[-1] = total_samples - sum(target_sizes[:-1])
    
    print("\nTarget split sizes:")
    for split_name, target in zip(['Train', 'Val', 'Test'], target_sizes):
        print(f"{split_name}: {target} samples ({target/total_samples:.2%})")
    
    # Analyze bin and geometry abundance
    bin_counts = Counter(bins)
    geometry_counts = {hash_: len(indices) for hash_, indices in geometry_groups.items()}
    
    # Define abundance thresholds
    bin_abundance_threshold = np.mean(list(bin_counts.values())) * 1.5
    geo_abundance_threshold = np.mean(list(geometry_counts.values())) * 1.5
    
    abundant_bins = {bin_: count for bin_, count in bin_counts.items() 
                    if count > bin_abundance_threshold}
    abundant_geometries = {hash_: count for hash_, count in geometry_counts.items() 
                         if count > geo_abundance_threshold}
    
    print(f"\nFound {len(abundant_bins)} abundant bins and {len(abundant_geometries)} abundant geometries")
    
    # Print abundance analysis
    print("\nBin distribution:")
    for bin_val, count in sorted(bin_counts.items()):
        status = "abundant" if bin_val in abundant_bins else "normal"
        print(f"Bin {bin_val}: {count} samples ({count/total_samples:.2%}) - {status}")
    
    print("\nGeometry distribution:")
    for hash_, count in sorted(geometry_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        status = "abundant" if hash_ in abundant_geometries else "normal"
        print(f"Geometry {hash_[:8]}...: {count} samples ({count/total_samples:.2%}) - {status}")
    
    # First pass: Initialize splits and ensure training set representation
    split_indices = [[] for _ in lengths]
    processed_geometries = set()
    remaining_indices = set(range(len(dataset)))
    
    # First ensure training set gets at least one sample from each bin
    for bin_val in bin_counts.keys():
        bin_samples = [idx for idx in range(len(dataset)) if bins[idx] == bin_val]
        if bin_samples:
            selected_idx = np.random.choice(bin_samples)
            split_indices[0].append(selected_idx)
            remaining_indices.remove(selected_idx)
    
    # Then ensure training set gets at least one sample from each geometry
    for hash_, group_indices in geometry_groups.items():
        available_indices = list(set(group_indices) & remaining_indices)
        if available_indices:
            selected_idx = np.random.choice(available_indices)
            split_indices[0].append(selected_idx)
            remaining_indices.remove(selected_idx)
    
    # Calculate remaining targets
    current_sizes = [len(indices) for indices in split_indices]
    remaining_targets = [target - current for target, current in zip(target_sizes, current_sizes)]
    
    # Second pass: Handle non-abundant cases
    for hash_, group_indices in geometry_groups.items():
        if hash_ not in abundant_geometries:
            available_indices = list(set(group_indices) & remaining_indices)
            if not available_indices:
                continue
                
            # Calculate adjusted lengths for this group
            group_total = len(available_indices)
            adjusted_lengths = [max(0, target)/sum(remaining_targets) 
                              for target in remaining_targets]
            
            group_splits = split_geometry_group(available_indices, bins, 
                                             adjusted_lengths, is_abundant=False)
            
            # Add splits while respecting remaining targets
            for split_idx, indices in enumerate(group_splits):
                to_add = min(len(indices), remaining_targets[split_idx])
                if to_add > 0:
                    selected_indices = list(indices)[:to_add]
                    split_indices[split_idx].extend(selected_indices)
                    remaining_indices.difference_update(selected_indices)
                    remaining_targets[split_idx] -= to_add
            
            processed_geometries.add(hash_)
    
    # Third pass: Redistribute abundant cases
    for hash_, group_indices in geometry_groups.items():
        if hash_ in abundant_geometries:
            available_indices = list(set(group_indices) & remaining_indices)
            if not available_indices or sum(remaining_targets) <= 0:
                continue
                
            # Calculate adjusted lengths for remaining samples
            adjusted_lengths = [max(0, target)/sum(remaining_targets) 
                              for target in remaining_targets]
            
            group_splits = split_geometry_group(available_indices, bins,
                                             adjusted_lengths, is_abundant=True)
            
            # Add splits while respecting remaining targets
            for split_idx, indices in enumerate(group_splits):
                to_add = min(len(indices), remaining_targets[split_idx])
                if to_add > 0:
                    selected_indices = list(indices)[:to_add]
                    split_indices[split_idx].extend(selected_indices)
                    remaining_indices.difference_update(selected_indices)
                    remaining_targets[split_idx] -= to_add
    
    # Final pass: Distribute any remaining samples to meet targets
    remaining_indices = list(remaining_indices)
    np.random.shuffle(remaining_indices)
    
    for idx in remaining_indices:
        # Find split with largest remaining target
        split_idx = np.argmax(remaining_targets)
        if remaining_targets[split_idx] > 0:
            split_indices[split_idx].append(idx)
            remaining_targets[split_idx] -= 1
    
    # Create final splits
    splits = [Subset(dataset, sorted(indices)) for indices in split_indices]
    
    # Print final statistics
    print("\nFinal Split Statistics:")
    for name, indices in zip(['Train', 'Val', 'Test'], split_indices):
        # Count unique geometries and bins in this split
        geometries_in_split = set()
        bins_in_split = set()
        
        for idx in indices:
            bins_in_split.add(bins[idx])
            for hash_, group_indices in geometry_groups.items():
                if idx in group_indices:
                    geometries_in_split.add(hash_)
        
        print(f"\n{name} Set:")
        print(f"Number of samples: {len(indices)} ({len(indices)/total_samples:.2%})")
        print(f"Number of unique geometries: {len(geometries_in_split)}")
        print(f"Number of unique bins: {len(bins_in_split)}")
    
    # Create visualization of final splits
    if log_dir:
        plot_split_analysis(splits, dataset, prediction_type, log_dir)
    
    # Verify splits
    verify_splits(splits, dataset, prediction_type)
    
    return splits

def verify_splits(splits, dataset, prediction_type):
    """Verify the quality of dataset splits."""
    total_samples = len(dataset)
    split_sizes = [len(split) for split in splits]
    split_ratios = [size/total_samples for size in split_sizes]
    
    print("\nSplit Verification:")
    print("Split sizes:", split_sizes)
    print("Split ratios:", [f"{ratio:.2%}" for ratio in split_ratios])
    
    # Verify value distributions
    if prediction_type == "buckling":
        values = [data.y.item() for data in dataset]
        split_values = [[values[i] for i in split.indices] for split in splits]
        
        print("\nValue distribution statistics:")
        for name, vals in zip(['Train', 'Val', 'Test'], split_values):
            print(f"\n{name} Set:")
            print(f"Mean: {np.mean(vals):.2e}")
            print(f"Std: {np.std(vals):.2e}")
            print(f"Min: {np.min(vals):.2e}")
            print(f"Max: {np.max(vals):.2e}")
    
    return True