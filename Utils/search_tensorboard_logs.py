import os
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

def find_tensorboard_logs(root_dir='.'):
    """
    Recursively search for TensorBoard log directories containing event files.
    
    Args:
        root_dir (str): Starting directory for the search
        
    Returns:
        list: List of tuples containing (tensorboard_dir, checkpoint_path)
    """
    tensorboard_dirs = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Look for TensorBoard event files
        if any(f.startswith('events.out.tfevents') for f in filenames):
            # Check for corresponding checkpoint
            checkpoint_path = os.path.join(dirpath, 'weights', 'last.pt')
            if os.path.exists(checkpoint_path):
                tensorboard_dirs.append((dirpath, checkpoint_path))
    
    return tensorboard_dirs

def load_tensorboard_data(log_dir):
    """
    Load TensorBoard events into a list of dictionaries containing metric data.
    
    Args:
        log_dir (str): Directory containing TensorBoard event files
        
    Returns:
        dict: Dictionary containing metric events
    """
    try:
        # Load the TensorBoard event file
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        # Get all scalar tags (metrics)
        tags = event_acc.Tags()['scalars']
        
        # Create a dictionary to store all metrics
        data = {}
        
        for tag in tags:
            # Get scalar events for this tag
            events = event_acc.Scalars(tag)
            # Store the events directly - they contain step and value attributes
            data[tag] = events
            
        return data
    
    except Exception as e:
        print(f"Error loading TensorBoard data from {log_dir}: {str(e)}")
        return {}

def load_checkpoint_config(checkpoint_path):
    """
    Load configuration from checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        dict: Dictionary containing config parameters
    """
    try:
        # Load checkpoint on CPU to save memory
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        
        # Convert all values to strings to ensure consistency
        return {k: str(v) for k, v in config.items()}
    
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
        return {}

def process_all_logs(root_dir='.'):
    """
    Process all TensorBoard logs and their configs into consolidated DataFrames.
    
    Args:
        root_dir (str): Root directory to search for TensorBoard logs. Defaults to current directory.
    
    Returns:
        tuple: (tensorboard_df, config_df) containing all processed data. Returns (None, None) if no valid logs found.
    """
    # Find all TensorBoard directories and their corresponding checkpoints in the specified root directory
    tensorboard_dirs = find_tensorboard_logs(root_dir)
    
    if not tensorboard_dirs:
        print(f"No TensorBoard logs found with corresponding checkpoints in directory: {root_dir}")
        return None, None
    
    print(f"\nFound {len(tensorboard_dirs)} valid log directories in {root_dir}")
    
    # Lists to store all data
    all_tensorboard_data = {}
    all_config_data = {}
    
    # Process each directory
    for tb_dir, checkpoint_path in tensorboard_dirs:
        print(f"\nProcessing directory: {tb_dir}")
        run_name = Path(tb_dir).name
        
        # Load TensorBoard data
        tb_data = load_tensorboard_data(tb_dir)
        
        # Load and process config
        config = load_checkpoint_config(checkpoint_path)
        config['run_name'] = run_name
        config['tensorboard_dir'] = os.path.abspath(tb_dir)
        config['checkpoint_path'] = os.path.abspath(checkpoint_path)
        prediction_type=config.get('prediction_type','no_config')
        if prediction_type in all_config_data:
            all_config_data[prediction_type].append(config)
        else:
            all_config_data[prediction_type] = [config]
        
        if not prediction_type in all_tensorboard_data:
            all_tensorboard_data[prediction_type] = []        
        # Process each metric
        for metric_name, events in tb_data.items():
            for event in events:
                all_tensorboard_data[prediction_type].append({
                    'run_name': run_name,
                    'metric': metric_name,
                    'step': event.step,
                    'value': event.value,
                    'wall_time': event.wall_time,
                    'tensorboard_dir': os.path.abspath(tb_dir)
                })

        print(f"Processed steps for run: {run_name}")
    
    # Create DataFrames
    tensorboard_df = {k:pd.DataFrame(v) for k,v in all_tensorboard_data.items()}
    config_df = {k:pd.DataFrame(v) for k,v in all_config_data.items()}
    
    return tensorboard_df, config_df

def save_dataframes(tensorboard_df_dict, config_df_dict, output_dir="analysis_output"):
    """
    Save the consolidated DataFrames using an optimized structure for later plotting.
    
    This function organizes the data in a way that makes it efficient to load and plot
    loss graphs later, even with large amounts of data. It creates separate files for
    different metrics and includes an index file for quick lookup.
    
    Args:
        tensorboard_df (pd.DataFrame): DataFrame containing all TensorBoard data
        config_df (pd.DataFrame): DataFrame containing all config data
        output_dir (str): Directory to save the output files
    """
    import json  # Add import at the top
    os.makedirs(output_dir, exist_ok=True)
    
    for prediction_type in config_df_dict.keys():
        tensorboard_df = tensorboard_df_dict[prediction_type]
        config_df = config_df_dict[prediction_type]

        pred_output_dir = os.path.join(output_dir, prediction_type)
        os.makedirs(pred_output_dir, exist_ok=True)

        # Create an index of all runs with their metadata
        run_index = {}
        
        # Get all unique metrics first
        metrics = tensorboard_df['metric'].unique()
        
        # Process each metric separately
        for metric in metrics:
            # Create a safe directory name by replacing invalid characters
            safe_metric_name = metric.replace('/', '_').replace('\\', '_')
            metric_dir = os.path.join(pred_output_dir, f"metric_{safe_metric_name}")
            os.makedirs(metric_dir, exist_ok=True)
            
            # Get data for this metric
            metric_data = tensorboard_df[tensorboard_df['metric'] == metric]
            
            # Group by run
            for run_name in metric_data['run_name'].unique():
                run_data = metric_data[metric_data['run_name'] == run_name]
                
                # Save run data efficiently
                run_file = os.path.join(metric_dir, f"{run_name}.pkl")
                run_data_dict = {
                    'steps': run_data['step'].values,
                    'values': run_data['value'].values,
                    'wall_time': run_data['wall_time'].values
                }
                
                # Save with high compression for storage efficiency
                with open(run_file, 'wb') as f:
                    torch.save(run_data_dict, f, pickle_protocol=5)
                
                # Initialize run in index if not exists
                if run_name not in run_index:
                    config_row = config_df[config_df['run_name'] == run_name]
                    if not config_row.empty:
                        config_dict = config_row.iloc[0].to_dict()
                    else:
                        config_dict = {}  # Empty dict if no config found
                    
                    run_index[run_name] = {
                        'metrics': [],  # Initialize as list directly
                        'config_row': config_dict
                    }
                
                # Append metric to the list if not already present
                if metric not in run_index[run_name]['metrics']:
                    run_index[run_name]['metrics'].append(metric)
        
        # Save the index as both JSON (for human readability) and pickle (for speed)
        with open(os.path.join(pred_output_dir, 'run_index.json'), 'w') as f:
            json.dump(run_index, f, indent=2)
        
        with open(os.path.join(pred_output_dir, 'run_index.pkl'), 'wb') as f:
            torch.save(run_index, f, pickle_protocol=5)
        
        # Print summary of saved data
        print("\nData saved successfully:")
        print(f"- Total runs processed: {len(run_index)}")
        print(f"- Total metrics found: {len(metrics)}")
        print("\nDirectory structure created:")
        print(f"- {pred_output_dir}/")
        print("  ├── run_index.json  # Human-readable index of all runs")
        print("  ├── run_index.pkl   # Fast-loading index for processing")
        for metric in metrics[:2]:  # Show first two metrics as example
            safe_metric = metric.replace('/', '_').replace('\\', '_')
            print(f"  ├── metric_{safe_metric}/")
        if len(metrics) > 2:
            print("  └── ... other metrics")

def main():
    """
    Main function to process and save all TensorBoard logs and checkpoints.
    Searches for logs in predefined root directories and processes all found TensorBoard data.
    """
    # Define the root directories to search for TensorBoard logs
    # We can add multiple paths to search in different locations
    root_dir = r"F:\Thesis\RESULTS"   
    
    print("Starting TensorBoard log analysis...")
    # Process logs from each root directory
    print(f"Processing logs in: {root_dir}")
    tensorboard_df, config_df = process_all_logs(root_dir)

    if tensorboard_df is not None and config_df is not None:
        # Save the consolidated data
        save_dataframes(tensorboard_df, config_df)
        
        # Print summary statistics
        print("\END")
        # print(f"Total number of runs processed: {len(config_df)}")
        # print(f"Total number of metric entries: {len(tensorboard_df)}")
        # print(f"Unique metrics found: {tensorboard_df['metric'].nunique()}")
        
    else:
        print("No data was processed. Please check if the directory contains valid TensorBoard logs.")

if __name__ == "__main__":
    main()