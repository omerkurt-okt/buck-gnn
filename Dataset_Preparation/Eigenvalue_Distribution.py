import os
import shutil
import numpy as np
from pyNastran.op2.op2 import OP2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
import csv
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import tempfile

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
    plt.rcParams['legend.fontsize'] = 16
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

def create_custom_bins(max_eigenvalue):
    """Create custom bin edges based on the specified ranges."""
    bins = []
    # 0.05 scaling up to 5
    bins.extend(np.arange(0, max_eigenvalue, 0.05))
    bins.append(max_eigenvalue)
    bins.append(float('inf'))
    return np.array(bins)

def extract_eigenvalue(op2_path):
    """Extract the minimum eigenvalue from an OP2 file."""
    try:
        op2 = OP2(debug=False)
        op2.read_op2(op2_path)
        
        if not hasattr(op2, 'eigenvectors') or not op2.eigenvectors:
            return op2_path, None
            
        buck_isubcase = list(op2.eigenvectors.keys())[0]
        eigenvectors = op2.eigenvectors[buck_isubcase]
        
        if not hasattr(eigenvectors, 'eigrs') or eigenvectors.eigrs is None:
            return op2_path, None

        eigenvalues = eigenvectors.eigrs
        return op2_path, np.min(eigenvalues)
    
    except Exception as e:
        print(f"Error processing {op2_path}: {str(e)}")
        return op2_path, None

# Constants for SSD cache
SSD_CACHE_DIR = r"C:\temp\op2_cache"  # Main cache directory on SSD
BATCH_SIZE = 1000  # Number of files to process in each batch

class OptimizedOP2Reader:
    def __init__(self, data_dir: str, ssd_cache_dir: str = SSD_CACHE_DIR, batch_size: int = BATCH_SIZE):
        """
        Initialize the optimized OP2 reader with SSD caching.
        
        Args:
            data_dir: Source directory containing OP2 files (HDD)
            ssd_cache_dir: Cache directory on SSD
            batch_size: Number of files to process in each batch
        """
        self.data_dir = data_dir
        self.ssd_cache_dir = ssd_cache_dir
        self.batch_size = batch_size
        self.temp_dir = os.path.join(ssd_cache_dir, 'temp_processing')
        self.cache_file = os.path.join(ssd_cache_dir, 'eigenvalue_cache.csv')
        self.cache_file = r"D:\Projects_Omer\DLNet\GNN_Project\ScreenShots\WO_Stiffener\Dataset\op2_cache\eigenvalue_cache.csv"
        
        # Create necessary directories
        os.makedirs(self.ssd_cache_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize thread-safe queues and locks
        self.file_queue = Queue(maxsize=batch_size)
        self.result_queue = Queue()
        self.copy_lock = threading.Lock()

    def _load_cache(self) -> Dict[str, float]:
        """Load existing cache if available."""
        cache = {}
        if os.path.exists(self.cache_file):
            print("Loading from cache...")
            try:
                df = pd.read_csv(self.cache_file)
                return dict(zip(df['filename'], df['eigenvalue']))
            except:
                print("Error reading cache file, starting fresh")
        return cache

    def _save_to_cache(self, new_data: Dict[str, float]):
        """Save new results to cache."""
        df = pd.DataFrame([
            {'filename': k, 'eigenvalue': v}
            for k, v in new_data.items()
        ])
        df.to_csv(self.cache_file, mode='a', header=not os.path.exists(self.cache_file), index=False)

    def _copy_to_ssd(self, filename: str) -> str:
        """Copy a file to SSD cache and return the new path."""
        source_path = os.path.join(self.data_dir, filename)
        temp_path = os.path.join(self.temp_dir, filename)
        
        with self.copy_lock:
            if not os.path.exists(temp_path):
                shutil.copy2(source_path, temp_path)
        
        return temp_path

    def _process_file(self, filename: str) -> Tuple[str, Optional[float]]:
        """Process a single file from SSD cache."""
        try:
            # Copy to SSD
            ssd_path = self._copy_to_ssd(filename)
            
            # Process file
            op2 = OP2(debug=False)
            op2.read_op2(ssd_path)
            
            if not hasattr(op2, 'eigenvectors') or not op2.eigenvectors:
                return filename, None
                
            buck_isubcase = list(op2.eigenvectors.keys())[0]
            eigenvectors = op2.eigenvectors[buck_isubcase]
            
            if not hasattr(eigenvectors, 'eigrs') or eigenvectors.eigrs is None:
                return filename, None

            eigenvalue = np.min(eigenvectors.eigrs)
            
            # Clean up
            os.remove(ssd_path)
            
            return filename, eigenvalue
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return filename, None

    def process_files(self) -> Tuple[np.ndarray, Dict]:
        """Process OP2 files with SSD caching and batching."""
        # Get list of all OP2 files
        op2_files = [f for f in os.listdir(self.data_dir) if f.endswith(".op2")]
        if not op2_files:
            raise ValueError(f"No .op2 files found in directory: {self.data_dir}")

        # Load existing cache
        cache = self._load_cache()
        print(f"Found {len(cache)} cached results")

        # Identify files that need processing
        files_to_process = [f for f in op2_files if f not in cache]
        print(f"Need to process {len(files_to_process)} new files")

        # Process new files in batches
        n_cores = max(1, cpu_count() -14)
        eigenvalues = []
        file_mapping = {}

        for i in range(0, len(files_to_process), self.batch_size):
            batch = files_to_process[i:i + self.batch_size]
            print(f"\nProcessing batch {i//self.batch_size + 1}/{(len(files_to_process) + self.batch_size - 1)//self.batch_size}")
            
            with ThreadPoolExecutor(max_workers=n_cores) as executor:
                batch_results = list(tqdm(
                    executor.map(self._process_file, batch),
                    total=len(batch),
                    desc="Processing files"
                ))

            # Update cache and results
            new_cache_entries = {}
            for filename, eigenvalue in batch_results:
                if eigenvalue is not None:
                    new_cache_entries[filename] = eigenvalue
                    eigenvalues.append(eigenvalue)
                    file_mapping[eigenvalue] = {
                        'op2': os.path.join(self.data_dir, filename),
                        'bdf': os.path.join(self.data_dir, filename.replace(".op2", ".bdf"))
                    }

            # Save batch results to cache
            self._save_to_cache(new_cache_entries)

        # Add cached results to output
        for filename, eigenvalue in cache.items():
            eigenvalues.append(eigenvalue)
            file_mapping[eigenvalue] = {
                'op2': os.path.join(self.data_dir, filename),
                'bdf': os.path.join(self.data_dir, filename.replace(".op2", ".bdf"))
            }

        # Clean up temp directory
        shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        return np.array(eigenvalues), file_mapping

def process_files_parallel(data_dir: str, cache_dir: str = SSD_CACHE_DIR) -> Tuple[np.ndarray, Dict]:
    """Enhanced parallel processing function with SSD caching."""
    reader = OptimizedOP2Reader(data_dir, cache_dir)
    return reader.process_files()

def plot_distribution_analysis(original_eigenvalues, bin_edges, output_dir):
    """Create initial distribution analysis plots and return statistics."""
    set_publication_style()
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])
    
    # First subplot: Linear scale
    ax1 = plt.subplot(gs[0])
    hist, bins, patches = ax1.hist(original_eigenvalues, bins=bin_edges, 
                                 alpha=0.8, color='#2C3E50',
                                 edgecolor='white', linewidth=0.5)
    
    ax1.set_title('Eigenvalue Distribution Analysis (Linear Scale)', 
                 pad=20, fontweight='bold')
    ax1.set_xlabel('Eigenvalue')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax1.set_xlim(left=0)  # Only set lower limit to 0

    # Second subplot: Log scale
    ax2 = plt.subplot(gs[1])
    ax2.hist(original_eigenvalues, bins=bin_edges, 
             alpha=0.8, color='#2C3E50',
             edgecolor='white', linewidth=0.5)
    
    ax2.set_title('Eigenvalue Distribution Analysis (Log Scale)', 
                 pad=20, fontweight='bold')
    ax2.set_xlabel('Eigenvalue')
    ax2.set_ylabel('Frequency')
    ax2.set_xscale('log')
    ax2.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'initial_distribution.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'initial_distribution.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    
    # Calculate bin statistics
    bin_stats = {}
    for i in range(len(bin_edges)-1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        mask = (original_eigenvalues >= bin_start) & (original_eigenvalues < bin_end)
        bin_values = original_eigenvalues[mask]
        
        bin_stats[i] = {
            'range': f'{bin_start:.3f}-{bin_end:.3f}',
            'count': len(bin_values),
            'min': float(np.min(bin_values)) if len(bin_values) > 0 else None,
            'max': float(np.max(bin_values)) if len(bin_values) > 0 else None,
            'mean': float(np.mean(bin_values)) if len(bin_values) > 0 else None
        }
    
    return hist, bin_stats

def create_all_distribution_plots(original_eigenvalues, selected_eigenvalues, bin_edges, output_dir):
    """Create all distribution plots with improved visibility."""
    set_publication_style()
    
    # Professional color scheme
    original_color = '#2C3E50'  # Dark blue-gray
    selected_color = '#E74C3C'  # Professional red
    
    # Common text for stats
    # stats_text = (f'Initial samples: {len(original_eigenvalues):,}\n'
    #              f'Final samples: {len(selected_eigenvalues):,}\n'
    #              f'Reduction ratio: {len(selected_eigenvalues)/len(original_eigenvalues):.1%}')
    stats_text = (f'Before 2.5% outlier removal: 200,000\n'
                 f'Initial samples: {len(original_eigenvalues):,}\n'
                 f'Final samples: 40,000\n'
                 f'Reduction ratio: {len(selected_eigenvalues)/len(original_eigenvalues):.1%}')

    # 1. Side by Side Comparison (Two separate plots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Initial distribution
    ax1.hist(original_eigenvalues, bins=bin_edges, 
            alpha=0.8, color=original_color,
            edgecolor='white', linewidth=0.5)
    ax1.set_title('Initial Distribution')
    ax1.set_xlabel('Eigenvalue')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(left=0)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Final distribution (with adjusted y-axis)
    final_hist, _, _ = ax2.hist(selected_eigenvalues, bins=bin_edges, 
                               alpha=0.8, color=selected_color,
                               edgecolor='white', linewidth=0.5)
    ax2.set_title('Final Distribution')
    ax2.set_xlabel('Eigenvalue')
    ax2.set_ylabel('Frequency')
    ax2.set_xlim(left=0)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust y-axis limits for final distribution
    max_final_height = np.max(final_hist)
    ax2.set_ylim(0, max_final_height * 1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'side_by_side_comparison.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'side_by_side_comparison.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 2. Overlaid with Secondary Axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot initial distribution on primary axis
    ax1.hist(original_eigenvalues, bins=bin_edges, alpha=0.4,
             label='Initial Distribution', color=original_color,
             edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Eigenvalue')
    ax1.set_ylabel('Initial Distribution Frequency', color=original_color)
    ax1.tick_params(axis='y', labelcolor=original_color)
    
    # Create secondary axis for final distribution
    ax2 = ax1.twinx()
    ax2.hist(selected_eigenvalues, bins=bin_edges, alpha=0.8,
             label='Final Distribution', color=selected_color,
             edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Final Distribution Frequency', color=selected_color)
    ax2.tick_params(axis='y', labelcolor=selected_color)
    
    plt.title('Comparison with Separate Scales')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dual_axis_comparison.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'dual_axis_comparison.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 3. Normalized Comparison (Percentage)
    fig = plt.figure(figsize=(12, 8))
    
    # Calculate normalized histograms
    hist1, _ = np.histogram(original_eigenvalues, bins=bin_edges)
    hist2, _ = np.histogram(selected_eigenvalues, bins=bin_edges)
    
    normalized_hist1 = hist1 / np.max(hist1)
    normalized_hist2 = hist2 / np.max(hist2)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.plot(bin_centers, normalized_hist1, color=original_color, 
             alpha=0.6, label='Initial Distribution', linewidth=2)
    plt.plot(bin_centers, normalized_hist2, color=selected_color, 
             alpha=0.8, label='Final Distribution', linewidth=2)
    
    plt.title('Normalized Distribution Comparison')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_comparison.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'normalized_comparison.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 1. Initial Distribution - Linear Scale
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    hist, bins, patches = ax.hist(original_eigenvalues, bins=bin_edges, 
                               alpha=0.8, color=original_color,
                               edgecolor='white', linewidth=0.5)
    
    ax.set_title('Initial Eigenvalue Distribution (Linear Scale)', 
               pad=20, fontweight='bold')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim(left=0)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'initial_distribution_linear.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'initial_distribution_linear.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    # 1a. Initial Distribution - Linear Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    hist, bins, patches = ax.hist(original_eigenvalues, bins=bin_edges, 
                               alpha=0.8, color=original_color,
                               edgecolor='white', linewidth=0.5)
    
    ax.set_title('Initial Eigenvalue Distribution (Linear Scale)', 
               pad=20, fontweight='bold')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim(left=0)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'initial_distribution_linear_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'initial_distribution_linear_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    # 2. Initial Distribution - Log Scale
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.hist(original_eigenvalues, bins=bin_edges, 
           alpha=0.8, color=original_color,
           edgecolor='white', linewidth=0.5)
    
    ax.set_title('Initial Eigenvalue Distribution (Log Scale)', 
               pad=20, fontweight='bold')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'initial_distribution_log.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'initial_distribution_log.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    # 2a. Initial Distribution - Log Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    ax.hist(original_eigenvalues, bins=bin_edges, 
           alpha=0.8, color=original_color,
           edgecolor='white', linewidth=0.5)
    
    ax.set_title('Initial Eigenvalue Distribution (Log Scale)', 
               pad=20, fontweight='bold')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'initial_distribution_log_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'initial_distribution_log_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    # 3. Final Distribution - Linear Scale
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    hist, bins, patches = ax.hist(selected_eigenvalues, bins=bin_edges, 
                               alpha=0.8, color=selected_color,
                               edgecolor='white', linewidth=0.5)
    
    ax.set_title('Final Eigenvalue Distribution (Linear Scale)', 
               pad=20, fontweight='bold')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim(left=0)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'final_distribution_linear.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'final_distribution_linear.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    # 3a. Final Distribution - Linear Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    hist, bins, patches = ax.hist(selected_eigenvalues, bins=bin_edges, 
                               alpha=0.8, color=selected_color,
                               edgecolor='white', linewidth=0.5)
    
    ax.set_title('Final Eigenvalue Distribution (Linear Scale)', 
               pad=20, fontweight='bold')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_xlim(left=0)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'final_distribution_linear_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'final_distribution_linear_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    # 4. Final Distribution - Log Scale
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.hist(selected_eigenvalues, bins=bin_edges, 
           alpha=0.8, color=selected_color,
           edgecolor='white', linewidth=0.5)
    
    ax.set_title('Final Eigenvalue Distribution (Log Scale)', 
               pad=20, fontweight='bold')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'final_distribution_log.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'final_distribution_log.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    # 4a. Final Distribution - Log Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    ax.hist(selected_eigenvalues, bins=bin_edges, 
           alpha=0.8, color=selected_color,
           edgecolor='white', linewidth=0.5)
    
    ax.set_title('Final Eigenvalue Distribution (Log Scale)', 
               pad=20, fontweight='bold')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'final_distribution_log_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'final_distribution_log_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    # 5. Comparison Plot - Linear Scale (Regular)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    plt.hist(original_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Initial Distribution', color=original_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(selected_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Final Distribution', color=selected_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Initial and Final Eigenvalue Distributions (Linear Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.xlim(left=0)
    
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
             fontsize=16)
    
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_linear.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_linear.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 5a. Comparison Plot - Linear Scale (Regular)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_ylim(top=5000)
    plt.hist(original_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Initial Distribution', color=original_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(selected_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Final Distribution', color=selected_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Initial and Final Eigenvalue Distributions (Linear Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.xlim(left=0)
    
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
             fontsize=16)
    
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_linear_constrained.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_linear_constrained.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()


    # 6. Comparison Plot - Linear Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    
    plt.hist(original_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Initial Distribution', color=original_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(selected_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Final Distribution', color=selected_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Initial and Final Eigenvalue Distributions (Linear Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.xlim(left=0)
    
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
             fontsize=16)
    
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_linear_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_linear_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 6a. Comparison Plot - Linear Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    ax.set_ylim(top=5000)
    plt.hist(original_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Initial Distribution', color=original_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(selected_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Final Distribution', color=selected_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Initial and Final Eigenvalue Distributions (Linear Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.xlim(left=0)
    
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
             fontsize=16)
    
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_linear_wide_constrained.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_linear_wide_constrained.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 7. Comparison Plot - Log Scale (Regular)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    plt.hist(original_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Initial Distribution', color=original_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(selected_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Final Distribution', color=selected_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Initial and Final Eigenvalue Distributions (Log Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    
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
             fontsize=16)
    
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_log.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_log.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 8. Comparison Plot - Log Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    
    plt.hist(original_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Initial Distribution', color=original_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(selected_eigenvalues, bins=bin_edges, alpha=0.6,
             label='Final Distribution', color=selected_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Initial and Final Eigenvalue Distributions (Log Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    
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
             fontsize=16)
    
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_log_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(output_dir, 'distribution_comparison_log_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

def calculate_samples_per_bin(bin_stats: dict, samples_per_bin: int) -> dict:
    """
    Take exactly specified number of samples from each bin if available.
    
    Args:
        bin_stats (dict): Statistics for each bin including current sample counts
        samples_per_bin (int): Number of samples to take from each bin
        
    Returns:
        dict: Dictionary with bin indices as keys and target sample counts as values
    """
    samples_per_bin_dict = {}
    total_selected = 0
    
    # Process each bin
    for bin_idx, stats in bin_stats.items():
        if stats['count'] > 0:
            # Take either samples_per_bin or all available samples if less
            samples_to_take = min(samples_per_bin, stats['count'])
            samples_per_bin_dict[bin_idx] = samples_to_take
            total_selected += samples_to_take
    
    print("\nDistribution statistics:")
    print(f"Number of populated bins: {len(samples_per_bin_dict)}")
    print(f"Samples per bin target: {samples_per_bin}")
    print(f"Total samples to be selected: {total_selected}")
    
    return samples_per_bin_dict

def flatten_distribution(data_dir: str, output_dir: str, samples_per_bin: int) -> None:
    """Process distribution with a target total sample count, excluding top and bottom 10% of eigenvalues."""
    print("Starting distribution analysis...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files in parallel
    eigenvalues, file_mapping = process_files_parallel(data_dir)

    # Calculate percentiles for filtering
    lower_bound = np.percentile(eigenvalues, 2.5)
    upper_bound = np.percentile(eigenvalues, 97.5)
    
    # Filter eigenvalues and update file mapping
    mask = (eigenvalues >= lower_bound) & (eigenvalues <= upper_bound)
    filtered_eigenvalues = eigenvalues[mask]
    
    print(f"\nFiltering statistics:")
    print(f"Original dataset size: {len(eigenvalues):,}")
    print(f"After removing top/bottom 2.5%: {len(filtered_eigenvalues):,}")
    print(f"Lower bound (2.5th percentile): {lower_bound:.4f}")
    print(f"Upper bound (97.5th percentile): {upper_bound:.4f}")
    
    # Create filtered file mapping
    filtered_file_mapping = {
        eigenvalue: file_mapping[eigenvalue]
        for eigenvalue in filtered_eigenvalues
    }

    # Calculate statistics for binning
    max_eigenvalue = np.max(filtered_eigenvalues)
    print(f"\nFiltered dataset eigenvalue statistics:")
    print(f"Maximum eigenvalue: {max_eigenvalue:.4f}")
    
    # Create bins using filtered data
    bin_edges = create_custom_bins(int(np.ceil(upper_bound)))
    
    # Create initial visualizations and get bin statistics
    hist, bin_stats = plot_distribution_analysis(filtered_eigenvalues, bin_edges, output_dir)
    
    # Calculate samples per bin based on target total
    samples_per_bin = calculate_samples_per_bin(bin_stats, samples_per_bin)
    
    # Print bin statistics
    print("\nCurrent bin statistics and target samples:")
    for bin_idx, stats in bin_stats.items():
        if stats['count'] > 0:
            target = samples_per_bin.get(bin_idx, 0)
            print(f"\nBin {bin_idx} ({stats['range']}):")
            print(f"  Current count: {stats['count']}")
            print(f"  Target count: {target}")
    
    # Select samples based on calculated targets
    selected_eigenvalues = []
    np.random.seed(42)
    
    for i in range(len(bin_edges)-1):
        if i not in samples_per_bin:
            continue
            
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        mask = (filtered_eigenvalues >= bin_start) & (filtered_eigenvalues < bin_end)
        bin_eigenvalues = filtered_eigenvalues[mask]
        
        if len(bin_eigenvalues) > 0:
            target_samples = samples_per_bin[i]
            bin_eigenvalues = np.random.permutation(bin_eigenvalues)
            selected_indices = np.random.choice(len(bin_eigenvalues), 
                                             target_samples, 
                                             replace=False)
            selected_eigenvalues.extend(bin_eigenvalues[selected_indices])
    
    selected_eigenvalues = np.array(selected_eigenvalues)
    
    # Create output directories and move files
    selected_dir = os.path.join(output_dir, "selected")
    remaining_dir = os.path.join(output_dir, "remaining")
    excluded_dir = os.path.join(output_dir, "excluded")  # New directory for excluded files
    
    for directory in [selected_dir, remaining_dir, excluded_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # print("\nCopying selected files...")
    # copied_files = 0
    # for eigenvalue in tqdm(selected_eigenvalues, desc="Copying files"):
    #     files = file_mapping[eigenvalue]
    #     for file_type in ['op2', 'bdf']:
    #         if os.path.exists(files[file_type]):
    #             dest_path = os.path.join(selected_dir, os.path.basename(files[file_type]))
    #             shutil.copy2(files[file_type], dest_path)
    #             copied_files += 1
    # print(f"\nSuccessfully copied {copied_files} files to {selected_dir}")

    # Create all visualizations
    create_all_distribution_plots(filtered_eigenvalues, selected_eigenvalues, bin_edges, output_dir)
    
    print(f"\nFiltering statistics:")
    print(f"Original dataset size: {len(eigenvalues):,}")
    print(f"After removing top/bottom 5%: {len(filtered_eigenvalues):,}")
    print(f"Lower bound (5th percentile): {lower_bound:.4f}")
    print(f"Upper bound (95th percentile): {upper_bound:.4f}")
    # Print summary
    print("\nProcess completed!")
    print(f"Original samples: {len(eigenvalues):,}")
    print(f"After filtering (2.5-97.5 percentile): {len(filtered_eigenvalues):,}")
    print(f"Selected samples: {len(selected_eigenvalues):,}")
    print(f"Final reduction ratio: {len(selected_eigenvalues)/len(eigenvalues):.2%}")

if __name__ == "__main__":
    data_dir = r"E:\WO_Stiffener"
    output_dir = r"D:\Projects_Omer\DLNet\GNN_Project\ScreenShots\WO_Stiffener\Dataset"
    samples_per_bin = 1040  # Specify desired total samples
    
    flatten_distribution(data_dir, output_dir, samples_per_bin)

# if __name__ == "__main__":
#     data_dir = r"E:\Dataset\W_Stiffener"
#     output_dir = r"D:\Projects_Omer\DLNet\GNN_Project\0_Data\W_Stiffener_MEGA"
#     max_samples_per_bin = 10
    
#     flatten_distribution(data_dir, output_dir, max_samples_per_bin)