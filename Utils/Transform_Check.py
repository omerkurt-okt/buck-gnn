import os
import numpy as np
import matplotlib.pyplot as plt
from pyNastran.bdf.bdf import BDF
import random
from typing import List
from Dataset_Preparation.GraphCreate import create_graph_from_bdf,parse_nastran_results
from typing import List, Tuple

def get_feature_names(prediction_type: str, use_z_coord: bool = False, 
                     use_rotations: bool = False, use_gp_forces: bool = False, 
                     use_super_node: bool = False) -> List[str]:
    """Get list of feature names based on model configuration"""
    feature_names = []
    
    # Coordinates
    if use_z_coord:
        feature_names.extend(["X coord", "Y coord", "Z coord"])
    else:
        feature_names.extend(["X coord", "Y coord"])
    
    # Basic features
    feature_names.extend([
        "SPC",
        "Force X", "Force Y",
        "Boundary",
        "Stiff 0°/180°",
        "Stiff 45°/225°",
        "Stiff 90°/270°",
        "Stiff 135°/315°"
    ])
    
    if prediction_type == "buckling":
        # Add displacement features
        if use_z_coord:
            feature_names.extend(["Disp X", "Disp Y", "Disp Z"])
        else:
            feature_names.extend(["Disp X", "Disp Y"])
            
        # Add rotation features if used
        if use_rotations:
            if use_z_coord:
                feature_names.extend(["Rot X", "Rot Y", "Rot Z"])
            else:
                feature_names.extend(["Rot X", "Rot Y"])
        
        # Add stress features
        feature_names.extend(["σx", "σy", "τxy"])
        
        # Add GP forces if used
        if use_gp_forces:
            for i in range(4):
                feature_names.extend([f"GP Force Q{i+1} X", f"GP Force Q{i+1} Y"])
    
    # Add super node indicator if used
    if use_super_node:
        feature_names.append("Super Node Flag")
        
    return feature_names

def create_feature_text(feature_names: List[str], features: np.ndarray, 
                       title: str, is_super_node: bool = False) -> str:
    """Create formatted feature text for display"""
    feature_text = f"{title}:\n"
    # if is_super_node:
    #     feature_text += "SUPER NODE\n"
    for name, feat in zip(feature_names, features):
        if abs(feat) < 0.001:
            feature_text += f"{name}: 0.000\n"
        else:
            feature_text += f"{name}: {feat:.3f}\n"
    return feature_text

def create_edge_feature_text(start_idx: int, end_idx: int, 
                           edge_features: np.ndarray, is_virtual: bool = False) -> str:
    """Create formatted edge feature text"""
    edge_type = "Virtual" if is_virtual else ("Stiffener" if edge_features[0] > 0.5 else "Shell")
    virtuality= "Virtual" if is_virtual else "Real"

    text = f"Edge {start_idx}-{end_idx} Features:\n"
    text += f"Stiffener Flag: {edge_type} [{edge_features[0]:.3f}]\n"
    text += f"Normalized Length: {edge_features[1]:.3f}\n"
    text += f"Direction: [{edge_features[2]:.3f}, {edge_features[3]:.3f}]\n"
    if len(edge_features) == 5:
        text += f"Virtuality Flag: {virtuality} [{edge_features[4]:.3f}]\n"
    
    return text
def set_publication_style():
    """Set up the plotting style for high-quality publication figures."""
    plt.style.use('seaborn-v0_8-paper')

    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 22

    # Set line width
    plt.rcParams['lines.linewidth'] = 2.3

    # Set other visual parameters
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.edgecolor'] = 'black'  

    # Set figure background color
    plt.rcParams['figure.facecolor'] = '#FFFFFF'
    plt.rcParams['axes.facecolor'] = '#FFFFFF'
    
def plot_edges(ax, node_features, edge_index, edge_features):
    """Helper function to plot edges consistently across all plots"""
    for i in range(edge_index.shape[1]):
        start_idx = edge_index[0, i]
        end_idx = edge_index[1, i]
        start_pos = node_features[start_idx, :2]
        end_pos = node_features[end_idx, :2]
        
        if edge_features[i, -1] == 1:  # Virtual edge
            color = '#2ECC71'
            alpha = 0.5
            linewidth = 1
            zorder = 1
        else:
            if edge_features[i, 0] > 0.5:  # Stiffener
                color = '#E74C3C'
                linewidth = 2
            else:  # Shell element
                color = '#F1C40F'
                linewidth = 1
            alpha = 0.7
            zorder = 2
        
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
    return ax
        
def plot_node_comparison(node_features_orig: np.ndarray, node_features_trans: np.ndarray,
                        edge_index_orig: np.ndarray, edge_index_trans: np.ndarray,
                        edge_features_orig: np.ndarray, edge_features_trans: np.ndarray,
                        highlighted_node: int, feature_names: List[str],
                        output_path: str, prediction_type: str) -> None:
    """Create comparison plot focusing on regular nodes and their features"""
    fig = plt.figure(figsize=(15, 7))
    
    # Plot settings
    set_publication_style()
    
    for subplot_idx, (node_features, edge_index, edge_features, title) in enumerate([
        (node_features_orig, edge_index_orig, edge_features_orig, "Original Graph - Node Features"),
        (node_features_trans, edge_index_trans, edge_features_trans, "Transformed Graph - Node Features")
    ]):
        ax = plt.subplot(1, 2, subplot_idx + 1)
        
        # Plot edges first
        ax = plot_edges(ax, node_features, edge_index, edge_features)
        
        is_super_node = node_features_orig[:, -1] == 1
        if not np.any(is_super_node):
            supernode=False
        else:
            supernode=True
        
        if supernode==True:
            super_idx = np.where(is_super_node)[0][0]
            # Plot regular nodes
            regular_nodes = ~is_super_node
                # Plot super node
            ax.scatter(node_features[super_idx, 0],
                    node_features[super_idx, 1],
                    c='#9B59B6', s=100, zorder=4, label='Super Node'
                    )
            ax.scatter(node_features[regular_nodes, 0],
                    node_features[regular_nodes, 1],
                    c='#3498DB', s=25, zorder=3, label='Regular Nodes')
        ax.scatter(node_features[:, 0],
                node_features[:, 1],
                c='#3498DB', s=25, zorder=3, label='Regular Nodes')
        
        # Highlight selected node
        ax.scatter(node_features[highlighted_node, 0],
                  node_features[highlighted_node, 1],
                  c='#E67E22', s=100, zorder=5, label='Highlighted Node')
        
        # Add node features in top left
        feature_text = create_feature_text(feature_names,
                                         node_features[highlighted_node],
                                         f"Node {highlighted_node} Features")
        
        ax.text(0.02, 0.98, feature_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white',
                         alpha=0.9,
                         edgecolor='gray',
                         linewidth=0.5,
                         boxstyle='round,pad=0.5'),
                fontsize=8)
        
        # Add legend in top right
        if supernode:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='#3498DB', label='Regular Node',
                        markersize=6, linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='#E67E22', label='Highlighted Node',
                        markersize=10, linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='#9B59B6', label='Super Node',
                        markersize=10, linestyle='None'),
                plt.Line2D([0], [0], color='#E74C3C', linewidth=2, label='Stiffener'),
                plt.Line2D([0], [0], color='#F1C40F', label='Shell Element'),
                plt.Line2D([0], [0], color='#2ECC71', label='Virtual Edge'),
            ]
        else:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='#3498DB', label='Nodes',
                        markersize=6, linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='#E67E22', label='Highlighted Node',
                        markersize=10, linestyle='None'),
                plt.Line2D([0], [0], color='#E74C3C', linewidth=2, label='Stiffener'),
                plt.Line2D([0], [0], color='#F1C40F', label='Shell Element'),
                plt.Line2D([0], [0], color='#2ECC71', label='Virtual Edge'),
            ]
        legend = ax.legend(handles=legend_elements, frameon=True,
                         facecolor='white', edgecolor='gray',
                         loc='upper right')
        legend.get_frame().set_linewidth(0.5)
        
        ax.set_title(title, pad=20, fontweight='bold')
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_nodes.png", dpi=300, bbox_inches='tight',
               facecolor='#FFFFFF', pad_inches=0.5)
    plt.close()

def plot_edge_comparison(node_features_orig: np.ndarray, node_features_trans: np.ndarray,
                        edge_index_orig: np.ndarray, edge_index_trans: np.ndarray,
                        edge_features_orig: np.ndarray, edge_features_trans: np.ndarray,
                        highlighted_edge: Tuple[int, int, int], output_path: str) -> None:
    """Create comparison plot focusing on edges and their features"""
    fig = plt.figure(figsize=(15, 7))
    
    # Plot settings
    set_publication_style()
    
    edge_idx, start_idx, end_idx = highlighted_edge
    
    for subplot_idx, (node_features, edge_index, edge_features, title) in enumerate([
        (node_features_orig, edge_index_orig, edge_features_orig, "Original Graph - Edge Features"),
        (node_features_trans, edge_index_trans, edge_features_trans, "Transformed Graph - Edge Features")
    ]):
        ax = plt.subplot(1, 2, subplot_idx + 1)
        
        # Plot all edges first
        ax = plot_edges(ax, node_features, edge_index, edge_features)
        
        # Highlight selected edge
        start_pos = node_features[start_idx, :2]
        end_pos = node_features[end_idx, :2]
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
               color='#E67E22', linewidth=3, zorder=6, label='Highlighted Edge')
        
        is_super_node = node_features_orig[:, -1] == 1
        if not np.any(is_super_node):
            supernode=False
        else:
            supernode=True
        
        if supernode==True:
            super_idx = np.where(is_super_node)[0][0]
            # Plot regular nodes
            regular_nodes = ~is_super_node
                # Plot super node
            ax.scatter(node_features[super_idx, 0],
                    node_features[super_idx, 1],
                    c='#9B59B6', s=100, zorder=4, label='Super Node'
                    )
            ax.scatter(node_features[regular_nodes, 0],
                    node_features[regular_nodes, 1],
                    c='#3498DB', s=25, zorder=3, label='Regular Nodes')
        ax.scatter(node_features[:, 0],
                node_features[:, 1],
                c='#3498DB', s=25, zorder=3, label='Regular Nodes')

        
        # Add edge features in top left
        edge_text = create_edge_feature_text(start_idx, end_idx,
                                           edge_features[edge_idx],
                                           is_virtual=False)
        
        ax.text(0.02, 0.98, edge_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(facecolor='white',
                        alpha=0.9,
                        edgecolor='gray',
                        linewidth=0.5,
                        boxstyle='round,pad=0.5'),
               fontsize=8)
        
        # Add legend in top right
        if supernode:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='#3498DB', label='Regular Nodes',
                        markersize=6, linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='#9B59B6', label='Super Node',
                        markersize=10, linestyle='None'),
                plt.Line2D([0], [0], color='#E67E22', linewidth=3, label='Highlighted Edge'),
                plt.Line2D([0], [0], color='#E74C3C', linewidth=2, label='Stiffener'),
                plt.Line2D([0], [0], color='#F1C40F', label='Shell Element'),
                plt.Line2D([0], [0], color='#2ECC71', label='Virtual Edge')
                
            ]
        else:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='#3498DB', label='Nodes',
                        markersize=6, linestyle='None'),
                plt.Line2D([0], [0], color='#E67E22', linewidth=3, label='Highlighted Edge'),
                plt.Line2D([0], [0], color='#E74C3C', linewidth=2, label='Stiffener'),
                plt.Line2D([0], [0], color='#F1C40F', label='Shell Element'),
                plt.Line2D([0], [0], color='#2ECC71', label='Virtual Edge'),
            ]
        legend = ax.legend(handles=legend_elements, frameon=True,
                         facecolor='white', edgecolor='gray',
                         loc='upper right')
        legend.get_frame().set_linewidth(0.5)
        
        ax.set_title(title, pad=20, fontweight='bold')
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_edges.png", dpi=300, bbox_inches='tight',
               facecolor='#FFFFFF', pad_inches=0.5)
    plt.close()

def plot_super_node_comparison(node_features_orig: np.ndarray, node_features_trans: np.ndarray,
                             edge_index_orig: np.ndarray, edge_index_trans: np.ndarray,
                             edge_features_orig: np.ndarray, edge_features_trans: np.ndarray,
                             feature_names: List[str], output_path: str) -> None:
    """Create comparison plot focusing on super node and its features"""
    # Only create if super node exists
    is_super_node = node_features_orig[:, -1] == 1
    if not np.any(is_super_node):
        return
    
    fig = plt.figure(figsize=(15, 7))
    
    # Plot settings
    set_publication_style()
    
    super_idx = np.where(is_super_node)[0][0]
    
    for subplot_idx, (node_features, edge_index, edge_features, title) in enumerate([
        (node_features_orig, edge_index_orig, edge_features_orig, "Original Graph - Super Node Features"),
        (node_features_trans, edge_index_trans, edge_features_trans, "Transformed Graph - Super Node Features")
    ]):
        ax = plt.subplot(1, 2, subplot_idx + 1)
        
        # Plot edges first
        ax = plot_edges(ax, node_features, edge_index, edge_features)
        
        # Plot regular nodes
        regular_nodes = ~is_super_node
        ax.scatter(node_features[regular_nodes, 0],
                  node_features[regular_nodes, 1],
                  c='#3498DB', s=25, zorder=3, label='Regular Nodes')
        
        # Plot super node
        ax.scatter(node_features[super_idx, 0],
                  node_features[super_idx, 1],
                  c='#9B59B6', s=100, zorder=4, label='Super Node'
                  )
        
        # Add super node features in top left
        feature_text = create_feature_text(feature_names,
                                         node_features[super_idx],
                                         f"Super Node Features",
                                         is_super_node=True)
        
        ax.text(0.02, 0.98, feature_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(facecolor='white',
                        alpha=0.9,
                        edgecolor='gray',
                        linewidth=0.5,
                        boxstyle='round,pad=0.5'),
               fontsize=8)
        
        # Add legend in top right
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='#3498DB', label='Regular Nodes',
                      markersize=6, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='#9B59B6', label='Super Node',
                      markersize=10, linestyle='None'),
            plt.Line2D([0], [0], color='#E74C3C', linewidth=2, label='Stiffener'),
            plt.Line2D([0], [0], color='#F1C40F', label='Shell Element'),
            plt.Line2D([0], [0], color='#2ECC71', label='Virtual Edge')
        ]
        legend = ax.legend(handles=legend_elements, frameon=True,
                         facecolor='white', edgecolor='gray',
                         loc='upper right')
        legend.get_frame().set_linewidth(0.5)
        
        ax.set_title(title, pad=20, fontweight='bold')
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_super_node.png", dpi=300, bbox_inches='tight',
               facecolor='#FFFFFF', pad_inches=0.5)
    plt.close()

# The main visualization function would then call these separate plotting functions
def visualize_graph_transformation(data_dir: str, num_samples: int = 5,
                                 use_z_coord: bool = False, use_rotations: bool = False,
                                 use_gp_forces: bool = False, use_axial_stress: bool = False,
                                 prediction_type: str = "buckling",
                                 use_super_node: bool = False,
                                 output_dir: str = 'transform_visualizations') -> None:
    """Create separate visualization plots for nodes, edges, and super nodes"""
    os.makedirs(output_dir, exist_ok=True)
    bdf_files = [f for f in os.listdir(data_dir) if f.endswith(".bdf")]
    
    for file_idx, bdf_file in enumerate(bdf_files[:num_samples]):
        print(f"\nProcessing file: {bdf_file}")
        
        # Load model
        bdf_model = BDF(debug=False)
        bdf_model.read_bdf(os.path.join(data_dir, bdf_file))
        
        # Get results
        op2_file = os.path.join(data_dir, bdf_file.replace('.bdf', '.op2'))
        eigenvalue, static_displacements, mode_shape, gp_stresses, gp_forces, cbar_stress = \
            parse_nastran_results(op2_file)
        
        # Create graphs for original and transformed
        node_features_orig, edge_index_orig, edge_features_orig, _, _, _, _ = create_graph_from_bdf(
            bdf_model, static_displacements, mode_shape, gp_forces, gp_stresses, cbar_stress,
            transform=False, use_z_coord=use_z_coord, use_rotations=use_rotations,
            use_gp_forces=use_gp_forces, use_axial_stress=use_axial_stress,
            use_mode_shapes_as_features=False, prediction_type=prediction_type,
            use_super_node=use_super_node, virtual_edge_cache=None
        )
        
        node_features_trans, edge_index_trans, edge_features_trans, _, _, _, _ = create_graph_from_bdf(
            bdf_model, static_displacements, mode_shape, gp_forces, gp_stresses, cbar_stress,
            transform=True, use_z_coord=use_z_coord, use_rotations=use_rotations,
            use_gp_forces=use_gp_forces, use_axial_stress=use_axial_stress,
            use_mode_shapes_as_features=False, prediction_type=prediction_type,
            use_super_node=use_super_node, virtual_edge_cache=None
        )
        
        # Convert to numpy
        node_features_orig = node_features_orig.numpy()
        node_features_trans = node_features_trans.numpy()
        edge_index_orig = edge_index_orig.numpy()
        edge_index_trans = edge_index_trans.numpy()
        edge_features_orig = edge_features_orig.numpy()
        edge_features_trans = edge_features_trans.numpy()
        
        # Get feature names for the current configuration
        feature_names = get_feature_names(
            prediction_type, use_z_coord, use_rotations,
            use_gp_forces, use_super_node
        )
        
        # Base filename for outputs
        filename_base = os.path.splitext(bdf_file)[0]
        output_path = os.path.join(output_dir, f'transform_visualization_{filename_base}')
        
        # Select random regular node to highlight
        if use_super_node:
            is_super_node = node_features_orig[:, -1] == 1
            regular_nodes = np.where(~is_super_node)[0]
        else:
            regular_nodes = np.arange(len(node_features_orig))
        
        highlighted_node = np.random.choice(regular_nodes)
        
        # Find a random non-virtual edge to highlight
        non_virtual_edges = []
        for i in range(edge_index_orig.shape[1]):
            if edge_features_orig[i, -1] != 1:  # Not a virtual edge
                start_idx = edge_index_orig[0, i]
                end_idx = edge_index_orig[1, i]
                non_virtual_edges.append((i, start_idx, end_idx))
        
        highlighted_edge = random.choice(non_virtual_edges) if non_virtual_edges else None
        
        # Create separate plots
        plot_node_comparison(
            node_features_orig, node_features_trans,
            edge_index_orig, edge_index_trans,
            edge_features_orig, edge_features_trans,
            highlighted_node, feature_names,
            output_path, prediction_type
        )
        
        if highlighted_edge:
            plot_edge_comparison(
                node_features_orig, node_features_trans,
                edge_index_orig, edge_index_trans,
                edge_features_orig, edge_features_trans,
                highlighted_edge, output_path
            )
        
        if use_super_node:
            plot_super_node_comparison(
                node_features_orig, node_features_trans,
                edge_index_orig, edge_index_trans,
                edge_features_orig, edge_features_trans,
                feature_names, output_path
            )
        
    print(f"\nVisualization complete. Files saved to: {output_dir}")


if __name__ == "__main__":
    visualize_graph_transformation(
        data_dir=r"D:\Projects_Omer\GNN_Project\ScreenShots\W_Stiffener\Transformation",
        num_samples=5,
        use_z_coord=False,
        use_rotations=False,
        use_gp_forces=False,
        use_axial_stress=False,
        prediction_type="buckling",
        use_super_node=False,
        output_dir=r"D:\Projects_Omer\GNN_Project\ScreenShots\W_Stiffener\TransformationSS"
    )