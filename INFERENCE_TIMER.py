import sys
sys.path.append(r"E:\Dataset\250301_Code_Database")
import os
from typing import  Optional
import logging
from dataclasses import dataclass
import shutil
from tqdm import tqdm
import subprocess
from copy import deepcopy
import time
from Dataset_Preparation.GraphCreate import load_single_data
import torch
from torch_geometric.loader import DataLoader
import multiprocessing as mp

CPU_COUNT_2_USE = 8

@dataclass
class Config:
    min_load: float
    max_load: float
    generate_stiffeners: bool = True
    min_active_stiffeners: int = 5
    max_active_stiffeners: int = 200
    min_consecutive: int = 5
    max_consecutive: int = 10
    loadcases_per_model: int = 10
    patterns_per_loadcase: int = 1000
    max_bc_lines: int = 3
    max_load_lines: int = 3
    max_nodes_per_line: int = 10
    min_nodes_per_line: int = 3
    max_nodes_per_load_line: int = 10
    min_nodes_per_load_line: int = 3
    direction_tolerance: float = 45.0  # degrees
    max_trials: int = 4
    eigenvalue_ratio_limit: float = 3.0
    high_ratio_acceptance_rate: float = 0.1  # 10%
    very_high_ratio_acceptance_rate: float = 0.05  # 5%
    nastran_path: str = r"C:\Program Files\MSC.Software\MSC_Nastran\2020sp1\bin\nastran.exe"
    temp_dir: str = "temp_analysis"
    delete_temp_files: bool = True

config = Config(min_load=0, max_load=0)
logger = logging.getLogger(__name__)

def run_nastran(bdf_path: str) -> Optional[str]:
    """Run Nastran analysis"""
    try:
        # Create a unique working directory for this analysis
        working_dir = os.path.dirname(bdf_path)
        os.makedirs(working_dir, exist_ok=True)
        
        bdf_path = os.path.abspath(bdf_path)
        op2_file = bdf_path.replace('.bdf', '.op2')
        
        # Copy the input file to working directory if it's not already there
        local_bdf = os.path.join(working_dir, os.path.basename(bdf_path))
        if local_bdf != bdf_path:
            shutil.copy2(bdf_path, local_bdf)
        
        cmd = [
            config.nastran_path,
            os.path.basename(local_bdf),  # Use basename since we're changing directory
            "scr=yes",
            "mem=2048"
        ]
        start = time.time()
        process = subprocess.run(
            cmd,
            cwd=working_dir,  # Set working directory for Nastran
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        end = time.time()
        if process.returncode != 0:
            logger.error(f"Nastran analysis failed: {process.stderr}")
            return -1
        
        return end - start
        
    except Exception as e:
        logger.error(f"Error running Nastran: {str(e)}")
        return None


def run_analysis(args):
    if isinstance(args, tuple) and len(args) == 2:
        cmd, working_dir = args
    else:
        print(f"Reading args")
        return None
    process = subprocess.run(
        cmd,
        cwd=working_dir,  # Set working directory for Nastran
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if process.returncode != 0:
        logger.error(f"Nastran analysis failed: {process.stderr}")
        print("Error in run_analysis\n")
        return -1


def run_nastran_parallel(bdf_path: str, batch_size) -> Optional[str]:
    """Run Nastran analysis"""
    try:
        # Create a unique working directory for this analysis
        working_dir = os.path.dirname(bdf_path)
        os.makedirs(working_dir, exist_ok=True)
        
        bdf_path = os.path.abspath(bdf_path)
        op2_file = bdf_path.replace('.bdf', '.op2')
        
        # Copy the input file to working directory if it's not already there
        local_bdf = os.path.join(working_dir, os.path.basename(bdf_path))
        if local_bdf != bdf_path:
            shutil.copy2(bdf_path, local_bdf)
        
        cmd = [
            config.nastran_path,
            os.path.basename(local_bdf),  # Use basename since we're changing directory
            "scr=yes",
            "mem=2048"
        ]

        process_args = [
            (cmd,working_dir)
            for _ in range(batch_size)]
        start = time.time()
        with mp.Pool(processes=CPU_COUNT_2_USE) as pool:
            result = list(pool.imap(run_analysis, process_args))
        end = time.time()

        result_problem = [res for res in result if res == -1]

        if len(result_problem) > 0:
            logger.error(f"Nastran analysis failed for {len(result_problem)} cases.")
            print(f"Nastran analysis failed for {len(result_problem)} cases.")
        
        return end - start
        
    except Exception as e:
        logger.error(f"Error running Nastran: {str(e)}")
        return None

def run_time_analysis(bdf_path: str, model_path: str, results_file: str, total_loop: int = 100, batch_size=128, device='cuda', NASTRAN=True) -> str:
    """Run time analysis for Nastran and inference"""

    total_nastran_parallel = 0
    total_nastran_single = 0
    total_GNN = 0
    run_nastran(bdf_path)
    if NASTRAN:
        run_nastran(bdf_path)
        print("Starting Nastran single inference loop...")
        for i in range(total_loop):
            infer_time=run_nastran(bdf_path)
            total_nastran_single = total_nastran_single + infer_time

        print("Starting Nastran parallel inference loop...")
        n_batch_size=batch_size
        for i in range(total_loop):
            infer_time=run_nastran_parallel(bdf_path, n_batch_size)
            total_nastran_parallel = total_nastran_parallel + infer_time

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    normalizer = checkpoint['normalizer']
    
    # Create model with the same configuration
    from Models.EA_GNN import EdgeAugmentedGNN
    model = EdgeAugmentedGNN(
        config['num_node_features'],
        config['num_edge_features'],
        config['hidden_channels'],
        config['num_layers'],
        config['pooling_layer'],
        prediction_type=config['prediction_type'],
        use_z_coord=config.get('use_z_coord', False),
        use_rotations=config.get('use_rotations', False),
        dropout_rate=config.get('dropout_rate', 0.1),
        model_name=config.get('model_name', 'EA_GNN')
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    process_args = [(
        bdf_path, 
        config['prediction_type'], 
        dict(
            use_z_coord=config['use_z_coord'],
            use_rotations=config['use_rotations'],
            use_gp_forces=False,
            use_axial_stress=False,
            use_mode_shapes_as_features=False,
            virtual_edges_file=None,
            use_super_node='super' in config['pooling_layer'],
        )
    ) for _ in range(1)]

    with mp.Pool(processes=1) as pool:
        results = list(tqdm(
            pool.imap(load_single_data, process_args),
            total=1,
            desc="Processing all files"
        ))
    dataset = [deepcopy(results[0]) for _ in range(batch_size)]

    # Prepare data based on prediction type
    if config['prediction_type'] == "static_stress":
        disp_dim = data[0].y.shape[1] - 3
        for data in data:
            data.y = data.y[:, disp_dim:]
    elif config['prediction_type'] == "static_disp":
        disp_dim = data[0].y.shape[1] - 3
        for data in data:
            data.y = data.y[:, :disp_dim]
    
    test_loader = DataLoader(dataset, batch_size=batch_size)

    print("Starting GNN inference loop...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            pred, adjusted_batch = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            for i in range(total_loop):
                start_time = time.time()
                pred, adjusted_batch = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                end_time = time.time()
                total_GNN = total_GNN + (end_time - start_time)
            break

    avg_GNN = total_GNN / total_loop
    if NASTRAN:
        avg_nastran_single = total_nastran_single / total_loop
        avg_nastran_parallel = total_nastran_parallel / total_loop

    print(f"Average GNN throughput: {batch_size/avg_GNN} samples/sec")
    if NASTRAN:
        print(f"Average NASTRAN single throughput: {1/avg_nastran_single} samples/sec")
        print(f"Average NASTRAN parallel throughput: {n_batch_size/avg_nastran_parallel} samples/sec")

    print(f"Average GNN batch latency: {avg_GNN/batch_size} seconds")
    if NASTRAN: 
        print(f"Average NASTRAN single latency: {avg_nastran_single/1} seconds")
        print(f"Aver    age NASTRAN parallel batch latency: {avg_nastran_parallel/n_batch_size} seconds")

    with open(results_file, 'w') as f:
        f.write(f"Model path: {model_path}\n")
        f.write(f"Batch size: {batch_size}, total loop: {total_loop}, cpu count: {CPU_COUNT_2_USE}\n\n")
        f.write(f"Average GNN throughput: {batch_size/avg_GNN} samples/sec\n")
        if NASTRAN:
            f.write(f"Average NASTRAN single throughput: {1/avg_nastran_single} samples/sec\n")
            f.write(f"Average NASTRAN parallel throughput: {n_batch_size/avg_nastran_parallel} samples/sec\n")
        f.write(f"Average GNN batch latency: {avg_GNN/batch_size} seconds\n")
        if NASTRAN:
            f.write(f"Average NASTRAN single latency: {avg_nastran_single/1} seconds\n")
            f.write(f"Average NASTRAN parallel latency: {avg_nastran_parallel/n_batch_size} samples/sec\n")

    if NASTRAN:
        return (avg_GNN/batch_size, avg_nastran_single/1, avg_nastran_parallel/n_batch_size)
    else:
        return (avg_GNN/batch_size)





def remove_files_with_stem_and_extensions(file_path):
    # Extract the stem (base name without extension) of the full file path
    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    directory = os.path.dirname(file_path)
    # List of extensions to remove
    extensions_to_remove = ['.op2', '.f04', '.f06', '.log']

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file ends with any of the specified extensions and starts with the same stem
        if any(filename.endswith(ext) for ext in extensions_to_remove) or any(ext+'.' in filename for ext in extensions_to_remove) and filename.startswith(file_stem):
            # Create the full file path for the file to remove
            file_to_remove = os.path.join(directory, filename)
            
            # Remove the file
            try:
                os.remove(file_to_remove)
            except Exception as e:
                pass

if __name__ == "__main__":
    
    NASTRAN = False
    BDF_PATH = r"E:\Dataset\inference_test_data\shape_w_cutout_0003_lc7_pristine.bdf"
    MODEL_PATH = r"E:\Dataset\inference_test_data\best.pt" 
    RESULTS_FILE = "inference_timing.txt"
    BATCH_SIZE = 128
    TOTAL_LOOP = 1
    timing_results = run_time_analysis(bdf_path=BDF_PATH, model_path=MODEL_PATH, results_file=RESULTS_FILE, total_loop=TOTAL_LOOP, batch_size=BATCH_SIZE, NASTRAN=NASTRAN)
    if NASTRAN:
        remove_files_with_stem_and_extensions(BDF_PATH)
    
