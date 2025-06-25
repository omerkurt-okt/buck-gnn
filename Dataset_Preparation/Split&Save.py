import os
import shutil
from collections import Counter
import numpy as np
from Dataset_Preparation.GraphCreate import load_dataset
from Dataset_Preparation.DatasetSplit import dataset_split
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import os

def set_publication_style():
    """Set up the plotting style."""
    plt.style.use('seaborn-v0_8-paper')

    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 18
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

def plot_split_distributions(train_data, val_data, save_dir):
    """Create professional publication-quality visualizations of dataset splits."""
    set_publication_style()
    
    # Define colors
    train_color = '#2C3E50'  # Dark blue-gray
    val_color = '#E74C3C'    # Professional red
    
    # Create statistics text
    stats_text = (f'Training samples: {len(train_data):,}\n'
                 f'Validation samples: {len(val_data):,}\n'
                 f'Split ratio: {len(train_data)/(len(train_data)+len(val_data)):.2%} / '
                 f'{len(val_data)/(len(train_data)+len(val_data)):.2%}')

    # 1. Individual Distributions - Linear Scale
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 2)
    
    # Training set
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(train_data, bins=50, alpha=0.8, color=train_color,
             edgecolor='white', linewidth=0.5)
    ax1.set_title('Training Set Distribution (Linear Scale)', pad=20)
    ax1.set_xlabel('Eigenvalue')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Validation set
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(val_data, bins=50, alpha=0.8, color=val_color,
             edgecolor='white', linewidth=0.5)
    ax2.set_title('Validation Set Distribution (Linear Scale)', pad=20)
    ax2.set_xlabel('Eigenvalue')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'split_distributions_linear.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'split_distributions_linear.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 2. Individual Distributions - Log Scale
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 2)
    
    # Training set
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(train_data, bins=50, alpha=0.8, color=train_color,
             edgecolor='white', linewidth=0.5)
    ax1.set_title('Training Set Distribution (Log Scale)', pad=20)
    ax1.set_xlabel('Eigenvalue')
    ax1.set_ylabel('Frequency')
    ax1.set_xscale('log')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Validation set
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(val_data, bins=50, alpha=0.8, color=val_color,
             edgecolor='white', linewidth=0.5)
    ax2.set_title('Validation Set Distribution (Log Scale)', pad=20)
    ax2.set_xlabel('Eigenvalue')
    ax2.set_ylabel('Frequency')
    ax2.set_xscale('log')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'split_distributions_log.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'split_distributions_log.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 3. Comparison Plot - Linear Scale
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    plt.hist(train_data, bins=50, alpha=0.6,
             label='Training Set', color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(val_data, bins=50, alpha=0.6,
             label='Validation Set', color=val_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Training and Validation Distributions (Linear Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    legend = plt.legend(loc='upper right', frameon=True, 
                       fancybox=True, framealpha=0.95, 
                       edgecolor='gray')
    legend.get_frame().set_linewidth(0.5)
    
    plt.text(0.98, 0.80, stats_text,
             transform=ax.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(facecolor='white', 
                      alpha=0.9,
                      edgecolor='gray',
                      linewidth=0.5,
                      boxstyle='round,pad=0.5'),
             fontsize=10)
    
    plt.savefig(os.path.join(save_dir, 'split_comparison_linear.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'split_comparison_linear.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 4. Comparison Plot - Log Scale
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    plt.hist(train_data, bins=50, alpha=0.6,
             label='Training Set', color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(val_data, bins=50, alpha=0.6,
             label='Validation Set', color=val_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Training and Validation Distributions (Log Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    legend = plt.legend(loc='upper right', frameon=True, 
                       fancybox=True, framealpha=0.95, 
                       edgecolor='gray')
    legend.get_frame().set_linewidth(0.5)
    
    plt.text(0.98, 0.80, stats_text,
             transform=ax.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(facecolor='white', 
                      alpha=0.9,
                      edgecolor='gray',
                      linewidth=0.5,
                      boxstyle='round,pad=0.5'),
             fontsize=10)
    
    plt.savefig(os.path.join(save_dir, 'split_comparison_log.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'split_comparison_log.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 5. Cumulative Distribution Comparison
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Sort data and calculate cumulative distributions
    train_sorted = np.sort(train_data)
    val_sorted = np.sort(val_data)
    train_cumulative = np.arange(1, len(train_sorted) + 1) / len(train_sorted)
    val_cumulative = np.arange(1, len(val_sorted) + 1) / len(val_sorted)
    
    plt.plot(train_sorted, train_cumulative, color=train_color, 
             label='Training Set', linewidth=2, alpha=0.8)
    plt.plot(val_sorted, val_cumulative, color=val_color, 
             label='Validation Set', linewidth=2, alpha=0.8)
    
    plt.title('Cumulative Distribution Comparison',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Cumulative Frequency')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    legend = plt.legend(loc='lower right', frameon=True, 
                       fancybox=True, framealpha=0.95, 
                       edgecolor='gray')
    legend.get_frame().set_linewidth(0.5)
    
    plt.text(0.98, 0.25, stats_text,
             transform=ax.transAxes,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(facecolor='white', 
                      alpha=0.9,
                      edgecolor='gray',
                      linewidth=0.5,
                      boxstyle='round,pad=0.5'),
             fontsize=10)
    
    plt.savefig(os.path.join(save_dir, 'cumulative_distribution.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'cumulative_distribution.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

def create_dataset_split_visualization(dataset, splits, save_dir):
    """Create comprehensive visualization for dataset splits."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract eigenvalues from dataset and splits
    train_data = np.array([dataset[i].y.item() for i in splits[0].indices])
    val_data = np.array([dataset[i].y.item() for i in splits[1].indices])
    
    # Create visualizations
    plot_split_distributions(train_data, val_data, save_dir)

    print("\nVisualization Statistics:")
    print(f"Training set: {len(train_data)} samples")
    print(f"  Mean: {np.mean(train_data):.4e}")
    print(f"  Std: {np.std(train_data):.4e}")
    print(f"  Min: {np.min(train_data):.4e}")
    print(f"  Max: {np.max(train_data):.4e}")
    print(f"\nValidation set: {len(val_data)} samples")
    print(f"  Mean: {np.mean(val_data):.4e}")
    print(f"  Std: {np.std(val_data):.4e}")
    print(f"  Min: {np.min(val_data):.4e}")
    print(f"  Max: {np.max(val_data):.4e}")

def dataset_split_folder_copy(dataset, splits, save_path):
    """
    Split dataset while maintaining distribution and target ratios.
    Saves the files to Train, Val, and Test folders.
    """

    os.makedirs(save_path, exist_ok=True)
    train_path = os.path.join(save_path, "Train")
    val_path = os.path.join(save_path, "Val")
    test_path = os.path.join(save_path, "Test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Save files to corresponding folders
    split_names = ["Train", "Val", "Test"]
    split_paths = [train_path, val_path, test_path]

    for split_name, dataset_subset, split_folder in zip(split_names, splits, split_paths):
        # Create a list containing only the data samples for this split
        split_data = [dataset_subset.dataset[i] for i in dataset_subset.indices]

        for data in dataset_subset:
            file_path = data.file_path
            if file_path:
                op2_file = file_path.replace(".bdf", ".op2")
                shutil.copy(file_path, os.path.join(split_folder, os.path.basename(file_path)))
                shutil.copy(op2_file, os.path.join(split_folder, os.path.basename(op2_file)))
            

        # Cache only this split's data
        dataset_cache_file = os.path.join(split_folder, f"dataset_cache_{prediction_type}.pkl")
        print(f"\nCaching {split_name} dataset ({len(split_data)} samples)...")
        with open(dataset_cache_file, "wb") as f:
            pickle.dump(split_data, f)

    print("\nFiles saved to:")
    print(f"Train folder: {train_path}")
    print(f"Val folder: {val_path}")
    print(f"Test folder: {test_path}")

# Replace original split logic with this approach.

if __name__ == '__main__':
    data_dir = r"D:\Projects_Omer\DLNet\GNN_Project\0_Data\W_Stiffener_MEGA\selected"
    save_path = r"D:\Projects_Omer\DLNet\GNN_Project\0_Data\Stiffener_Buckling_RandomEdge"
    vis_path = os.path.join(save_path, "visualizations")  # Add this line
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)  # Add this line
    
    use_z_coord = False
    use_rotations = False
    use_gp_forces = False
    use_axial_stress = False
    use_mode_shapes_as_features = False
    use_super_node = True
    prediction_type = "buckling"
    virtual_edges_file = None
    
    

    dataset, normalizer = load_dataset(
        data_dir, 
        use_z_coord=use_z_coord,
        use_rotations=use_rotations,
        use_gp_forces=use_gp_forces,
        use_axial_stress=use_axial_stress,
        use_mode_shapes_as_features=use_mode_shapes_as_features,
        virtual_edges_file=virtual_edges_file,
        use_super_node=use_super_node,
        prediction_type=prediction_type
    )

    normalizer_cache_file = os.path.join(save_path, f"normalizer_cache.pkl")
    print("\nNormalizing dataset...")
    with open(normalizer_cache_file, "wb") as f:
        pickle.dump(normalizer, f)

    if len(dataset) == 0:
        raise ValueError("No valid data was loaded. Please check input files.")

    splits = dataset_split(dataset=dataset,
        prediction_type=prediction_type, 
        lengths=[0.90, 0.10],
        remove_outliers=True,
        n_bins=1000,
        log_dir= save_path+"/logs"
        )
    
    create_dataset_split_visualization(dataset, splits, vis_path)

    dataset_split_folder_copy(dataset, splits, save_path)


