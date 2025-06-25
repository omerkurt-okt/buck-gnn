import sys
sys.path.append(r"E:\Dataset\250301_Code_Database")
import os
import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from Dataset_Preparation.GraphCreate import load_folder_dataset
from Dataset_Preparation.Metrics import MAPE_error, stress_errors
from collections import Counter
from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm

EXCEL_REPORT_DIR = r'D:\Projects_Omer\GNN_Project\REPORTS2.xlsx'

STATIC_METRICS = ['re', 'max_disp_rel', 'max_disp_mae']
BUCKLING_METRICS = ['MAPE', 'MIN MAPE', 'MAX MAPE']
CONFIG_KEYS = ['num_node_features', 'num_edge_features', 'hidden_channels', 'num_layers', 'use_edge_attr', 'use_z_coord', 'use_rotations', 'prediction_type', 'pooling_layer', 'dropout_rate', 'model_name']
EXCEL_COLUMNS = ['Weight Dir', 'Data Dir'] + CONFIG_KEYS + BUCKLING_METRICS + STATIC_METRICS


def update_excel_report(results, model_path, data_dir, config):
    prediction_type = config['prediction_type']
    if not os.path.exists(EXCEL_REPORT_DIR):
        df = pd.DataFrame(columns=EXCEL_COLUMNS)
    else:
        df = pd.read_excel(EXCEL_REPORT_DIR)
    
    new_row = {
        'Weight Dir': os.path.dirname(model_path),
        'Data Dir': data_dir
    }
    for key in CONFIG_KEYS:
        new_row[key] = config.get(key, None)
        
    for metric in BUCKLING_METRICS + STATIC_METRICS:
        new_row[metric] = None
    
    if prediction_type == "buckling":
        for metric in BUCKLING_METRICS:
            if metric in results:
                new_row[metric] = results[metric]
    else:
        for metric in STATIC_METRICS:
            if metric in results:
                new_row[metric] = results[metric]
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(EXCEL_REPORT_DIR, index=False)

def run_inference(model_path, test_data_dir, output_dir, batch_size=128, device='cuda', config_name=''):
    """
    Run inference on test data and report metrics to TensorBoard.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        test_data_dir (str): Directory containing test data
        output_dir (str): Directory for output files and TensorBoard logs
        batch_size (int): Batch size for inference
        device (str): Device to run inference on ('cuda' or 'cpu')
    """
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    normalizer = checkpoint['normalizer']
    
    # Create model with the same configuration

    from Models.BuckGNN import BuckGNN

    model = BuckGNN(
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
    
    # Load test dataset
    test_data = load_folder_dataset(
        test_data_dir,
        normalizer=normalizer,
        use_z_coord=config['use_z_coord'],
        use_rotations=config['use_rotations'],
        use_gp_forces=False,
        use_axial_stress=False,
        use_mode_shapes_as_features=False,
        virtual_edges_file=None,
        use_super_node='super' in config['pooling_layer'],
        prediction_type=config['prediction_type']
    )
    
    # Prepare data based on prediction type
    if config['prediction_type'] == "static_stress":
        disp_dim = test_data[0].y.shape[1] - 3
        for data in test_data:
            data.y = data.y[:, disp_dim:]
    elif config['prediction_type'] == "static_disp":
        disp_dim = test_data[0].y.shape[1] - 3
        for data in test_data:
            data.y = data.y[:, :disp_dim]
    
    test_loader = DataLoader(test_data, batch_size=batch_size)
    

    # Initialize metrics
    
    total_mape = 0
    max_mape = 0
    min_mape = 99999
    test_total_graphs = sum(1 for _ in test_loader.dataset)
    
    # Create results directory
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / 'inference_results.txt'

    # Set up TensorBoard writer
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=results_dir/ f'inference_{timestamp}')
     
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            batch = batch.to(device)
            pred, adjusted_batch = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Handle different prediction types
            if config['prediction_type'] == "buckling":
                # Log denormalized predictions for buckling
                true_vals = normalizer.denormalize_eigenvalue(batch.y)
                pred_vals = normalizer.denormalize_eigenvalue(pred)
                mapes = torch.abs((true_vals - pred_vals) / true_vals)
                mape = torch.sum(mapes).item() * 100
                total_mape += mape

                if mapes.max().item()*100 > max_mape:
                    max_mape = mapes.max().item()*100
                if mapes.min().item()*100 < min_mape:
                    min_mape = mapes.min().item()*100

                
            elif "static" in config['prediction_type']:       
                if config['prediction_type'] == "static_disp":
                    true_vals = normalizer.denormalize_displacement(batch.y)
                    pred_vals = normalizer.denormalize_displacement(pred)
                else:  # static_stress
                    true_vals = normalizer.denormalize_gp_stresses(batch.y)
                    pred_vals = normalizer.denormalize_gp_stresses(pred)
                
                error_dict = stress_errors(
                    pred_vals,
                    true_vals,
                    adjusted_batch,
                    config['prediction_type'],
                    threshold=0.0001 if config['prediction_type'] == "static_disp" else 0.2
                )
                
                if batch_idx == 0:
                    test_dict = error_dict
                else:
                    test_dict = dict(Counter(error_dict) + Counter(test_dict))
    
    # Calculate and log final metrics
    if config['prediction_type'] == "buckling":
        avg_mape = total_mape / test_total_graphs

        writer.add_scalar('MAPE/test', avg_mape, 0)
        writer.add_scalar('MAPE-min/test', min_mape, 0)
        writer.add_scalar('MAPE-max/test', max_mape, 0)


        with results_file.open('w') as f:
            f.write(f"Final Test MAPE: {avg_mape:.2f}%\n")
            f.write(f"Final Test Min MAPE: {min_mape:.2f}%\n")
            f.write(f"Final Test Max MAPE: {max_mape:.2f}%\n")

            
    elif "static" in config['prediction_type']:
        res_dict = {key: test_dict[key]/len(test_loader) for key in STATIC_METRICS}
        #test_dict = {key: value/test_total_graphs for key, value in test_dict.items()}
        
        for key, value in res_dict.items():
            writer.add_scalar(f'{key}/test', value, 0)
        
        with results_file.open('w') as f:
            f.write("Final Test Metrics:\n")
            for key, value in res_dict.items():
                f.write(f"{key}: {value:.4f}\n")

    if config['prediction_type'] == "buckling":
        results = {'MAPE' : avg_mape, 'MIN MAPE': min_mape, 'MAX MAPE': max_mape}
    else:
        results = res_dict
    
    update_excel_report(results, model_path, test_data_dir, config)
    writer.close()
    return results_dir

if __name__ == "__main__":
    # Configuration
    import argparse

    # # Set up argument parser
    # parser = argparse.ArgumentParser(description="Select model and dataset for testing.")

    # # Add arguments for model and data
    # parser.add_argument("--model", type=int, required=True, help="Model type: 1 for static, 3 for buckling.")
    # parser.add_argument("--data", type=int, required=True, help="Dataset: 1 for Shapes_100, 2 for Shapes_100_2_2, etc.")

    # # Parse the arguments
    # args = parser.parse_args()

    # # Assign arguments to variables
    # model = args.model
    model = 3
    # data = args.data
    data = 4
    # Static disp wo stiffener
    if model==1:
        MODEL_PATH = r"E:\Dataset\RESULTS\STATIC_w_stiffener\SN_GraphSAGE_128_6\OUTPUT_DIR\tensorboard_logs\91df4_00000\weights\last.pt"
        # MODEL_PATH = r"E:\Dataset\RESULTS\STATIC_wo_stiffener\SN_ADD_512_3_Robust_ReLUCorrection\OUTPUT_DIR\tensorboard_logs\c8023_00000\weights\last.pt"
    #    MODEL_PATH = r"E:\Dataset\RESULTS\STATIC_wo_stiffener\SN_ADD_512_3_Robust_ReLUCorrection\OUTPUT_DIR\tensorboard_logs\c8023_00000\weights\last.pt"
    elif model==2:
        MODEL_PATH = r"E:\Dataset\RESULTS\BUCKLING_wo_stiffener\512_3_SNONLY_ReLUCorrection\OUTPUT_DIR\tensorboard_logs\3c520_00000\weights\last.pt"
    elif model==3:
        MODEL_PATH = r"E:\Dataset\RESULTS\BUCKLING_w_stiffener\SN_GraphSAGE_128_6_Robust\OUTPUT_DIR\tensorboard_logs\20026_00000\weights\last.pt"


    if data==1:
        TEST_DATA_DIR = r"D:\Projects_Omer\DLNet\GNN_Project\0_Data\TEST\wo_stiffener\Shapes_100" 
    elif data==2:
        TEST_DATA_DIR = r"D:\Projects_Omer\DLNet\GNN_Project\0_Data\TEST\wo_stiffener\Shapes_100_2_2" 
    elif data==3:
        TEST_DATA_DIR = r"E:\Dataset\TEST\Beam" 
    elif data==4:
        TEST_DATA_DIR = r"E:\Dataset\TEST\Bulkhead" 

    
    OUTPUT_DIR = Path(MODEL_PATH).parent.parent 
    BATCH_SIZE = 1
    TRAINING_CONFIG_NAME = "test"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results_dir = run_inference(
        model_path=MODEL_PATH,
        test_data_dir=TEST_DATA_DIR,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        device=device
    )
    
    print(f"Inference complete. Results saved to: {results_dir}")