import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from matplotlib.gridspec import GridSpec



class MetricPlotter:
    """
    A class to parse and visualize TensorBoard metrics from our saved data structure.
    """
    @staticmethod
    def set_publication_style():
        """Set up the plotting style"""
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
        
    def __init__(self, analysis_dir):
        """
        Initialize the plotter with the directory containing the analyzed data.
        
        Args:
            analysis_dir (str): Path to the directory containing the processed TensorBoard data
        """
        self.analysis_dir = Path(analysis_dir)
        self.run_index = self._load_run_index()
        self.available_metrics = self._get_available_metrics()
        
        # Set up a consistent color palette for runs
        self.color_palette = sns.color_palette("husl", n_colors=len(self.run_index))
        self.run_colors = dict(zip(self.run_index.keys(), self.color_palette))
        
        # Set default plot style
        self.set_publication_style()
        
    def _load_run_index(self):
        """Load and return the run index from the pickle file."""
        index_path = self.analysis_dir / 'run_index.pkl'
        if not index_path.exists():
            raise FileNotFoundError(f"No run index found at {index_path}")
        
        with open(index_path, 'rb') as f:
            return torch.load(f)
    
    def _get_available_metrics(self):
        """Get all unique metrics from the run index."""
        metrics = set()
        for run_data in self.run_index.values():
            metrics.update(run_data['metrics'])
        return sorted(metrics)
    
    def _load_metric_data(self, run_name, metric):
        """Load metric data for a specific run."""
        safe_metric = metric.replace('/', '_').replace('\\', '_')
        metric_file = self.analysis_dir / f"metric_{safe_metric}" / f"{run_name}.pkl"
        
        if not metric_file.exists():
            return None
        
        with open(metric_file, 'rb') as f:
            return torch.load(f)
    
    def list_available_runs(self):
        """Print available runs and their metrics."""
        print("\nAvailable Runs and Metrics:")
        print("-" * 50)
        for run_name, run_data in self.run_index.items():
            config = run_data['config_row']
            print(f"\nRun: {run_name}")
            print("Metrics:", ', '.join(run_data['metrics']))
            print("Key Config Parameters:")
            for key in ['learning_rate', 'batch_size', 'optimizer']:
                if key in config:
                    print(f"  - {key}: {config[key]}")
    
    def plot_metrics(self, metric, runs=None, smooth_factor=0, figsize=(15, 6),
                    grid=True, legend_loc='best', title=None, xlim=None, ylim=None, ylabel=None):
        """
        Plot specified metrics for selected runs.
        
        Args:
            metrics (str or list): Metric(s) to plot
            runs (list, optional): List of run names to plot. If None, plots all runs
            smooth_factor (float): Moving average window for smoothing (0 for no smoothing)
            figsize (tuple): Figure size in inches
            grid (bool): Whether to show grid
            legend_loc (str): Legend location
            title (str): Plot title. If None, auto-generates based on metrics
        """
            
        # Validate metric
        if not metric in self.available_metrics:
            raise ValueError(f"Invalid metric: {metric}")
        
        # Setup runs to plot
        if runs is None:
            runs = list(self.run_index.keys())
        else:
            invalid_runs = [r for r in runs if r not in self.run_index]
            if invalid_runs:
                raise ValueError(f"Invalid runs: {invalid_runs}")
        
        # Create subplot grid
        fig = plt.figure(figsize=(figsize[0], figsize[1]))
        
        ax = fig.add_subplot(111)
        
        for run_name in runs:
            if metric in self.run_index[run_name]['metrics']:
                data = self._load_metric_data(run_name, metric)
                if data is None:
                    continue
                
                x = data['steps']
                y = data['values']
                
                # Apply smoothing if requested
                if smooth_factor > 0:
                    y = np.convolve(y, np.ones(smooth_factor)/smooth_factor, mode='valid')
                    x = x[smooth_factor-1:]
                
                if x.shape != y.shape:
                    continue
                # Create label with key configuration parameters
                config = self.run_index[run_name]['config_row']
                label = f"{run_name}"
                if 'learning_rate' in config:
                    label += f" (lr={config['learning_rate']})"
                
                ax.plot(x, y, label=label, color=self.run_colors[run_name])
        
        ax.set_xlabel('Epochs')
        if ylabel is None:
            ax.set_ylabel(metric)
        else:
            ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim([0,200])
        if title:
            ax.set_title(title, fontsize=14, y=1.02)
        
        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        # ax.legend(loc=legend_loc)
        
        # Set overall title if provided

        plt.tight_layout()
        return fig
    
    def plot_metric_comparison(self, metric, runs=None, aggregate='mean',
                             window_size=100, figsize=(12, 6)):
        """
        Create a comparative analysis plot for a metric across different runs.
        
        Args:
            metric (str): Metric to compare
            runs (list, optional): List of run names to compare. If None, uses all runs
            aggregate (str): Aggregation method ('mean' or 'median') for windowing
            window_size (int): Size of the window for aggregation
            figsize (tuple): Figure size in inches
        """
        if metric not in self.available_metrics:
            raise ValueError(f"Invalid metric: {metric}")
            
        if runs is None:
            runs = list(self.run_index.keys())
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Line plot with confidence intervals
        for run_name in runs:
            if metric in self.run_index[run_name]['metrics']:
                data = self._load_metric_data(run_name, metric)
                if data is None:
                    continue
                
                values = data['values']
                steps = data['steps']
                
                # Calculate rolling statistics
                rolling_mean = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                rolling_std = np.array([np.std(values[i:i+window_size]) 
                                     for i in range(len(values)-window_size+1)])
                
                x = steps[window_size-1:]
                if x.shape != rolling_mean.shape:
                    continue
                ax1.plot(x, rolling_mean, label=run_name, color=self.run_colors[run_name])
                ax1.fill_between(x, 
                               rolling_mean - rolling_std,
                               rolling_mean + rolling_std,
                               alpha=0.2,
                               color=self.run_colors[run_name])
        
        ax1.set_title(f"{metric} Progression")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel(metric)
        # ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Box plot of values
        data_to_plot = []
        labels = []
        
        for run_name in runs:
            if metric in self.run_index[run_name]['metrics']:
                data = self._load_metric_data(run_name, metric)
                if data is not None:
                    data_to_plot.append(data['values'])
                    labels.append(run_name)
        
        ax2.boxplot(data_to_plot, labels=labels)
        ax2.set_title(f"{metric} Distribution")
        ax2.set_ylabel(metric)
        plt.xticks(rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig

def main():
    path_plot = "analysis_output/buckling"
    metric =  'MAPE/val'
    title = 'Validation MAPE'
    
    # Initialize plotter
    plotter = MetricPlotter(path_plot)
    #['Learning_Rate', 'Loss/train', 'Loss/train_batch', 'Loss/validation', 'MAPE/train', 'MAPE/val']
    # Show available runs and metrics
    plotter.list_available_runs()
    
    # Example: Plot training and validation loss
    plotter.plot_metrics(metric, smooth_factor=0, title=title, #xlim=[0, 300], ylim=[0, 2000], 
                                            ylabel='MAPE', runs = ['e48c2_00000'])
    plt.savefig(os.path.join(path_plot, metric.replace('/','_')+'_metric_plot.png'))
    plt.show()
    
    
    # Example: Compare accuracy across runs
    plotter.plot_metric_comparison('MAPE/val')
    plt.savefig(os.path.join(path_plot, metric.replace('/','_')+'_metric_comparison.png'))

    plt.show()

if __name__ == "__main__":
    main()