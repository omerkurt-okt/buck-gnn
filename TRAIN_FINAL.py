import os
import sys
import time
import torch
from torch_geometric.loader import DataLoader
import logging
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray import train as t2
from ray.tune.logger import TBXLoggerCallback
from Utils.Losses import get_loss_function
from Models.EA_GNN import EdgeAugmentedGNN
from Dataset_Preparation.GraphCreate import load_folder_dataset
from Dataset_Preparation.Metrics import MAPE_error,stress_errors
import matplotlib
from pathlib import Path
import pickle
from collections import Counter
# matplotlib.use('Agg')  # Use non-interactive backend

# Constants
INP_DIR = r"D:\Projects_Omer\GNN_Project\0_Data\Dataset_Splitted_RandomEdge"
OUTPUT_DIR = r"D:\Projects_Omer\GNN_Project\OUTPUT_DIR"
RAY_RESULT_DIR = r"D:\ray_results"
# BASEDIR = r"/arf/scratch/omerkurt/Static_wo_stiffener/SN_ADD_128_6_Robust/"
# INP_DIR = BASEDIR + r"Dataset_Splitted"
# OUTPUT_DIR = BASEDIR + r"OUTPUT_DIR"
# RAY_RESULT_DIR = BASEDIR + r"ray_results"

USE_Z_COORD_GLOB = False
USE_ROT_GLOB = False
USE_AXIAL_STRESS_GLOB = False
USE_SUPER_NODE_GLOB = False
PREDICTION_TYPE_GLOB = "buckling"  # can be "buckling", "static_stress", "static_disp" or "mode_shape"
BATCH_SIZE_GLOB = 128
INITIAL_LR_GLOB = 1e-2

SLEEP_GLOB = 0      # Timer, 60*60*3.5 for 3.5 hours
CPU_PER_TRIAL = 20  # For ray-tune
GPU_PER_TRIAL = 1   # For ray-tune
GRACE_PERIOD_GLOB = 1500

SCHEDULER_GLOB = "cosine" # "cosine" "restart"
USE_LR_SCHEDULER_GLOB = True
T_0_GLOB = 500    # Initial restart interval (epochs)
T_M_GLOB = 2     # Multiplier for restart interval
MIN_LR_GLOB = INITIAL_LR_GLOB/100

MODE_GLOB = "manual"    # "manual" or "auto"

CONFIG_HYPERPARAMETER_GLOB = {
    "lr": INITIAL_LR_GLOB,
    "hidden_channels": 128,
    "num_layers": 6,
    "weight_decay": 1e-8,
    "num_epochs": 1501,
    "loss_function": tune.grid_search(["graph_rel", "graph_mae"]), # "relative_error", "graph_mse", "graph_mae", "graph_p90_rel", "graph_mixed", "graph_max_rel"
    "use_edge_attr": True,
    "pooling_layer": "mlp",             # "supernode_with_pooling" "supernode_only" "mlp" "mlp_no_super" /// IMPORTANT /// if "super" in self.pooling_layer then return self.decoder(x[is_real_node])
    "use_z_coord": USE_Z_COORD_GLOB,
    "use_rotations": USE_ROT_GLOB,
    # "dropout_rate": tune.grid_search([0.1, 0.2, 0.3, 0.4])
    "dropout_rate": 0.1,
    "model_name": "GraphSage_addAggr_Shared"          # "GraphSage_MLP", "GraphSage_meanAggr", "GraphSage_sumAggr", "GraphSage_sumAggr_woBatchNorm", "GraphSage_maxAggr", "EA_GNN", "EA_GNN_Shared"
}

CONFIG_MANUAL_GLOB = {
    "lr": INITIAL_LR_GLOB,
    "hidden_channels": 128,
    "num_layers": 6,
    "weight_decay": 1e-8,
    "num_epochs": 1501,
    "loss_function": "relative_error",     # buckling: "relative_error" static: "graph_mse", "graph_mae", "graph_p90_rel", "graph_mixed", "graph_max_rel"
    "use_edge_attr": True,
    "pooling_layer": "mean",             # "supernode_with_pooling" "supernode_only" "mlp" "mlp_no_super"
    "use_z_coord": USE_Z_COORD_GLOB,
    "use_rotations": USE_ROT_GLOB,
    "dropout_rate": 0.1,
    "model_name": "GraphSage_addAggr_Shared"            # "GraphSage_MLP", "GraphSage_addAggr", "GraphSage_sumAggr", "GraphSage_sumAggr_woBatchNorm", "GraphSage_maxAggr", "EA_GNN", "EA_GNN_Shared"
}

PLOT_DATABASE_INFO_GLOB=True
##########################################################################################################################################

# Check output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(filename='gnn_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Check for CUDA availability for manual mode
if MODE_GLOB == "manual":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

def hyperparameter_optimization(train_loader, val_loader, num_node_features, num_edge_features, prediction_type, normalizer):

    config = CONFIG_HYPERPARAMETER_GLOB
    if "static" in prediction_type:
        scheduler = ASHAScheduler(
            metric="Validation_Loss",
            mode="min",
            max_t=config["num_epochs"]-1,
            grace_period=GRACE_PERIOD_GLOB,
            reduction_factor=10)
        reporter = CLIReporter(
            metric_columns=["Train_Loss", "Train_MSE", "Validation_Loss", "Val_MSE", "training_iteration"])
    elif prediction_type == "buckling":
        scheduler = ASHAScheduler(
            metric="Val_MAPE",
            mode="min",
            max_t=config["num_epochs"]-1,
            grace_period=GRACE_PERIOD_GLOB,
            reduction_factor=4)
        reporter = CLIReporter(
            metric_columns=["Train_Loss", "Train_MAPE", "Validation_Loss", "Val_MAPE", "training_iteration"])


    result = tune.run(
        tune.with_parameters(train_gnn, data_loaders=(num_node_features, num_edge_features, train_loader, val_loader, allValues, prediction_type,normalizer)),
        resources_per_trial={"cpu": CPU_PER_TRIAL, "gpu": GPU_PER_TRIAL},
        config=config,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter,
        callbacks=[TBXLoggerCallback()],
        name="ea_gnn_tuning",
        storage_path=RAY_RESULT_DIR,
        trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
        resume=False
    )

    if "static" in prediction_type:
        best_trial = result.get_best_trial("Val_MSE", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation MSE: {}".format(
            best_trial.last_result["Val_MSE"]))
    elif prediction_type == "buckling":
        best_trial = result.get_best_trial("Val_MAPE", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation MAPE: {}".format(
            best_trial.last_result["Val_MAPE"]))

    return best_trial.config

def create_model(num_node_features, num_edge_features, hidden_channels=128, num_layers=6, 
                use_edge_attr=True, pooling_layer='mean', prediction_type="buckling",
                use_z_coord=False, use_rotations=False, dropout_rate=0.1, model_name="EA_GNN"):  # Added dropout_rate parameter
    print(f"Creating EA-GNN model with:")
    print(f"  num_node_features: {num_node_features}")
    print(f"  num_edge_features: {num_edge_features}")
    print(f"  hidden_channels: {hidden_channels}")
    print(f"  num_layers: {num_layers}")
    print(f"  use_edge_attr: {use_edge_attr}")
    print(f"  prediction_type: {prediction_type}")
    print(f"  dropout_rate: {dropout_rate}")
    print(f"  model_name: {model_name}")
    return EdgeAugmentedGNN(
        num_node_features, num_edge_features, hidden_channels, num_layers, 
        pooling_layer, prediction_type=prediction_type,
        use_z_coord=use_z_coord, use_rotations=use_rotations,
        dropout_rate=dropout_rate, model_name=model_name
    )
    
def train_gnn(config, checkpoint_dir=None, data_loaders=None, nosave = False):
    num_node_features, num_edge_features, train_loader, val_loader, allValues, prediction_type, normalizer= data_loaders

    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    print(f"train_gnn: Node features: {num_node_features}, Edge features: {num_edge_features}")
    model = create_model(
        num_node_features, num_edge_features, 
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        use_edge_attr=config["use_edge_attr"],
        prediction_type=prediction_type,
        pooling_layer=config["pooling_layer"],
        use_z_coord=config.get("use_z_coord", False),
        use_rotations=config.get("use_rotations", False),
        dropout_rate=config.get("dropout_rate", 0.1),
        model_name=config.get("model_name"),
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    if SCHEDULER_GLOB == "restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0_GLOB,
            T_mult=T_M_GLOB,
            eta_min=MIN_LR_GLOB  # Minimum learning rate
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=T_0_GLOB,
            # T_mult=T_M_GLOB,
            eta_min=MIN_LR_GLOB  # Minimum learning rate
        )
    criterion = get_loss_function(config["loss_function"], allValues, 
                                use_z_coord=config.get("use_z_coord", False),
                                use_rotations=config.get("use_rotations", False))

    try:
        trial_id = t2.get_context().get_trial_id()
        if trial_id is not None:
            ray_running=True
        else:
            ray_running=False
    except RuntimeError:
        trial_id = f"manual_run_{int(time.time())}"
    finally:
        if trial_id is None:
            trial_id = f"manual_run_{int(time.time())}"

    train_total_graphs = 0
    for batch in train_loader:
        train_total_graphs += batch.batch.max().item() + 1
    print(f"Total number of graphs: {train_total_graphs}")
    val_total_graphs = 0
    for batch in val_loader:
        val_total_graphs += batch.batch.max().item() + 1
    print(f"Total number of graphs: {val_total_graphs}")

    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, 'tensorboard_logs', trial_id))
    log_dir = Path(writer.log_dir)
    print(f"Log dir = {log_dir}")
    results_file = log_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write("trial_id: {trial_id}\n\n")
        for k, v in config.items():
            f.write(str(k) + ' : '+ str(v) + '\n\n')
 
    wdir = str(log_dir / 'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    best_fitness = 1e10

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0
        train_mape = 0

        current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            pred, adjusted_batch = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            if USE_SUPER_NODE_GLOB == True:
                is_real_node = batch.x[:, -1] == 0
                filtered_x = batch.x[is_real_node]
            else:
                filtered_x = batch.x
            if prediction_type == "buckling":
                loss = criterion(normalizer.denormalize_eigenvalue(pred), normalizer.denormalize_eigenvalue(batch.y))
                train_mape += MAPE_error(pred, batch.y, prediction_type, normalizer).item()
            elif prediction_type == "static_disp":
                loss = criterion(
                    normalizer.denormalize_displacement(pred), 
                    normalizer.denormalize_displacement(batch.y), 
                    adjusted_batch,
                    filtered_x
                )
                error_dict = stress_errors(
                    normalizer.denormalize_displacement(pred), 
                    normalizer.denormalize_displacement(batch.y),
                    adjusted_batch,
                    prediction_type,
                    threshold=0.0001
                )
            elif prediction_type == "static_stress":
                batch_stress = batch.y.contiguous()
                loss = criterion(
                    normalizer.denormalize_gp_stresses(pred), 
                    normalizer.denormalize_gp_stresses(batch_stress), 
                    adjusted_batch,
                    filtered_x
                )
                error_dict = stress_errors(
                    normalizer.denormalize_gp_stresses(pred), 
                    normalizer.denormalize_gp_stresses(batch_stress),
                    adjusted_batch,
                    prediction_type,
                    threshold=0.2
                )
            else:
                loss = criterion(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if "static" in prediction_type:
                if batch_idx == 0:
                    train_dict = error_dict
                else:
                    train_dict = dict(Counter(error_dict) + Counter(train_dict))


            if batch_idx % len(train_loader) == 0:
                writer.add_scalar('Loss/train_batch', train_loss, epoch * len(train_loader) + batch_idx)

        train_loss /= len(train_loader)
        if USE_LR_SCHEDULER_GLOB == True:
            scheduler.step()

        # Log the learning rate
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)

        if "static" in prediction_type:
            train_dict = {key: value/train_total_graphs for key, value in train_dict.items()}
            for key,value in train_dict.items():
                writer.add_scalar(f'{key}/train', value, epoch)
        else:
            train_mape /= len(train_loader)
            writer.add_scalar('MAPE/train', train_mape, epoch)
       
        model.eval()
        val_loss = 0
        val_mape = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = batch.to(device)
                pred, adjusted_batch = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                if USE_SUPER_NODE_GLOB == True:
                    is_real_node = batch.x[:, -1] == 0
                    filtered_x = batch.x[is_real_node]
                else:
                    filtered_x = batch.x
                if prediction_type == "buckling":
                    val_loss += criterion(normalizer.denormalize_eigenvalue(pred), normalizer.denormalize_eigenvalue(batch.y)).item()
                    val_mape += MAPE_error(pred, batch.y, prediction_type, normalizer).item()
                elif prediction_type == "static_disp":
                    val_loss += criterion(
                        normalizer.denormalize_displacement(pred), 
                        normalizer.denormalize_displacement(batch.y), 
                        adjusted_batch,
                        filtered_x
                    ).item()
                    error_dict = stress_errors(
                        normalizer.denormalize_displacement(pred), 
                        normalizer.denormalize_displacement(batch.y),
                        adjusted_batch,
                        prediction_type,
                        threshold=0.0001
                    )
                elif prediction_type == "static_stress":
                    batch_stress = batch.y.contiguous()
                    val_loss += criterion(
                        normalizer.denormalize_gp_stresses(pred), 
                        normalizer.denormalize_gp_stresses(batch_stress), 
                        adjusted_batch,
                        filtered_x
                    ).item()
                    error_dict = stress_errors(
                        normalizer.denormalize_gp_stresses(pred), 
                        normalizer.denormalize_gp_stresses(batch_stress),
                        adjusted_batch,
                        prediction_type,
                        threshold=0.2
                    )
                else:
                    val_loss += criterion(pred, batch.y).item()
                    
                if "static" in prediction_type:
                    if batch_idx == 0:
                        val_dict = error_dict
                    else:
                        val_dict = dict(Counter(error_dict) + Counter(val_dict))

        val_loss /= len(val_loader)
        
        writer.add_scalar('Loss/validation', val_loss, epoch)
        if "static" in prediction_type:
            val_dict = {key:value/val_total_graphs for key,value in val_dict.items()}
            for key,value in val_dict.items():
                writer.add_scalar(f'{key}/val', value, epoch)
        else:
            val_mape /= len(val_loader)
            writer.add_scalar('MAPE/val', val_mape, epoch)

        save = (not nosave) or (epoch == config["num_epochs"] - 1)
        if save:
            # Save last, best and delete old checkpoints
            torch.save({
                'model_state_dict': model.state_dict(),
                'normalizer': normalizer,
                'config': {
                    'num_node_features': num_node_features,
                    'num_edge_features': num_edge_features,
                    'hidden_channels': config["hidden_channels"],
                    'num_layers': config["num_layers"],
                    'use_edge_attr': config["use_edge_attr"],
                    'use_z_coord': use_z_coord,
                    'use_rotations': use_rotations,
                    'prediction_type': prediction_type,
                    'pooling_layer': config["pooling_layer"],
                    'dropout_rate': config["dropout_rate"],
                    'model_name': config["model_name"]
                }
            }, last)
            if prediction_type=="buckling" and val_mape < best_fitness:
                best_fitness = val_mape
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'normalizer': normalizer,
                    'config': {
                        'num_node_features': num_node_features,
                        'num_edge_features': num_edge_features,
                        'hidden_channels': config["hidden_channels"],
                        'num_layers': config["num_layers"],
                        'use_edge_attr': config["use_edge_attr"],
                        'use_z_coord': use_z_coord,
                        'use_rotations': use_rotations,
                        'prediction_type': prediction_type,
                        'pooling_layer': config["pooling_layer"],
                        'dropout_rate': config["dropout_rate"],
                        'model_name': config["model_name"]
                    }
                }, best)


        if "static" in prediction_type:
            s = f"Epoch {epoch+1}/{config['num_epochs']}, Train_Loss: {train_loss:.4f}, Val_Loss: {val_loss:.4f},"
            s = s + '\n'
            for key,value in train_dict.items():
                s = s + f'Train_{key}: {value:.4f}, '
            s = s + '\n'
            for key,value in val_dict.items():
                s = s + f'Val_{key}: {value:.4f}, '
        else:
            s = f"Epoch {epoch+1}/{config['num_epochs']}, Train_Loss: {train_loss:.4f}, Train_Mape: {train_mape:.2f}%, Val_Loss: {val_loss:.4f}, Val_Mape:{val_mape:.2f}%"
        
        with open(results_file, "a") as f:
            f.write(s+'\n')
        print(s)

        if ray_running==True:
            if "static" in prediction_type:
                val_mse = val_dict.get('mse', float('inf'))
                train_mse = train_dict.get('mse', float('inf'))
                t2.report({"Validation_Loss": val_loss, "Val_MSE": val_mse, "Train_Loss": train_loss, "Train_MSE": train_mse})
            elif prediction_type == "buckling":
                t2.report({"Validation_Loss": val_loss, "Val_MAPE": val_mape, "Train_Loss": train_loss, "Train_MAPE":train_mape})
    writer.close()
    return model

import matplotlib.pyplot as plt
import numpy as np
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

def visualize_split_distributions(train_eigenvalues, val_eigenvalues, save_dir):
    """Create comprehensive visualizations of dataset splits with publication quality."""
    os.makedirs(save_dir, exist_ok=True)
    set_publication_style()
    
    # Define colors
    train_color = '#2C3E50'  # Dark blue-gray
    val_color = '#E74C3C'    # Professional red
    
    # Create statistics text
    # stats_text = (f'Training samples: {len(train_eigenvalues):,}\n'
    #              f'Validation samples: {len(val_eigenvalues):,}\n'
    #              f'Split ratio: {len(train_eigenvalues)/(len(train_eigenvalues)+len(val_eigenvalues)):.2%} / '
    #              f'{len(val_eigenvalues)/(len(train_eigenvalues)+len(val_eigenvalues)):.2%}')
    stats_text = (f'Training samples: 36,000\n'
                 f'Validation samples: 4,000\n'
                 f'Split ratio: {0.9:.2%} / '
                 f'{0.1:.2%}')
    # min_val = min(min(train_eigenvalues), min(val_eigenvalues))
    min_val = min(train_eigenvalues)
    # max_val = max(max(train_eigenvalues), max(val_eigenvalues))
    max_val = max(train_eigenvalues)
    bins_calc = np.linspace(min_val, max_val, 51)  # 50 bins + 1 for edges
    # 1. Individual Distributions - Training Set (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.8, color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Training Set Distribution (Linear Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=min_val)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'training_distribution_linear.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'training_distribution_linear.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 1a. Individual Distributions - Training Set (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.8, color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Training Set Distribution (Linear Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=min_val)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'training_distribution_linear_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'training_distribution_linear_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 2. Individual Distributions - Training Set Log Scale (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.8, color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Training Set Distribution (Log Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'training_distribution_log.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'training_distribution_log.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 2a. Individual Distributions - Training Set Log Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.8, color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Training Set Distribution (Log Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'training_distribution_log_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'training_distribution_log_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 3. Individual Distributions - Validation Set (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.8, color=val_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Validation Set Distribution (Linear Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=min_val)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_linear.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_linear.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 3a. Individual Distributions - Validation Set (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.8, color=val_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Validation Set Distribution (Linear Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=min_val)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_linear_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_linear_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 4. Individual Distributions - Validation Set Log Scale (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.8, color=val_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Validation Set Distribution (Log Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_log.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_log.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 4a. Individual Distributions - Validation Set Log Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.8, color=val_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Validation Set Distribution (Log Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_log_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_log_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 5. Comparison Plot - Linear Scale (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Training Set', color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Validation Set', color=val_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Training and Validation Distributions (Linear Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=min_val)
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
             fontsize=18)
    
    plt.savefig(os.path.join(save_dir, 'comparison_linear.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'comparison_linear.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 5a. Comparison Plot - Linear Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Training Set', color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Validation Set', color=val_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Training and Validation Distributions (Linear Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=min_val)
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
             fontsize=18)
    
    plt.savefig(os.path.join(save_dir, 'comparison_linear_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'comparison_linear_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 6. Comparison Plot - Log Scale (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Training Set', color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Validation Set', color=val_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Training and Validation Distributions (Log Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.xlim(left=0)
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
             fontsize=18)
    
    plt.savefig(os.path.join(save_dir, 'comparison_log_normal.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'comparison_log_normal.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 6. Comparison Plot - Log Scale (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Training Set', color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Validation Set', color=val_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Training and Validation Distributions (Log Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.xlim(left=0)
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
             fontsize=18)
    
    plt.savefig(os.path.join(save_dir, 'comparison_log_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'comparison_log_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

def visualize_split_distributions2(train_eigenvalues, val_eigenvalues, save_dir):
    """Create comprehensive visualizations of dataset splits with publication quality."""
    os.makedirs(save_dir, exist_ok=True)
    set_publication_style()
    
    # Define colors
    train_color = '#2C3E50'  # Dark blue-gray
    val_color = '#E74C3C'    # Professional red
    print(len(train_eigenvalues))
    print(len(val_eigenvalues))
    # Create statistics text
    # stats_text = (f'Training samples: {len(train_eigenvalues):,}\n'
    #              f'Validation samples: {len(val_eigenvalues):,}\n'
    #              f'Split ratio: {len(train_eigenvalues)/(len(train_eigenvalues)+len(val_eigenvalues)):.2%} / '
    #              f'{len(val_eigenvalues)/(len(train_eigenvalues)+len(val_eigenvalues)):.2%}')
    stats_text = (f'Training samples: 34,000\n'
                 f'Validation samples: 6,000\n'
                 f'Split ratio: {0.9:.2%} / '
                 f'{0.1:.2%}')
    # min_val = min(min(train_eigenvalues), min(val_eigenvalues))
    min_val = 0
    # max_val = max(max(train_eigenvalues), max(val_eigenvalues))
    max_val = max(train_eigenvalues)
    bins_calc = np.linspace(min_val, max_val, 51)  # 50 bins + 1 for edges
    # 1. Individual Distributions - Training Set (Normal)
    # 1. Individual Distributions - Training Set (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.8, color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Training Set Distribution (Linear Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'training_distribution_linear.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'training_distribution_linear.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 1a. Individual Distributions - Training Set (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.8, color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Training Set Distribution (Linear Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'training_distribution_linear_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'training_distribution_linear_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 2. Individual Distributions - Training Set Log Scale (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.8, color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Training Set Distribution (Log Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'training_distribution_log.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'training_distribution_log.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 2a. Individual Distributions - Training Set Log Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.8, color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Training Set Distribution (Log Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'training_distribution_log_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'training_distribution_log_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 3. Individual Distributions - Validation Set (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.8, color=val_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Validation Set Distribution (Linear Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_linear.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_linear.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 3a. Individual Distributions - Validation Set (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.8, color=val_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Validation Set Distribution (Linear Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_linear_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_linear_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 4. Individual Distributions - Validation Set Log Scale (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.8, color=val_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Validation Set Distribution (Log Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_log.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_log.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 4a. Individual Distributions - Validation Set Log Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.8, color=val_color,
             edgecolor='white', linewidth=0.5)
    plt.title('Validation Set Distribution (Log Scale)', pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_log_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'validation_distribution_log_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 5. Comparison Plot - Linear Scale (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Training Set', color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Validation Set', color=val_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Training and Validation Distributions (Linear Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
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
             fontsize=18)
    
    plt.savefig(os.path.join(save_dir, 'comparison_linear.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'comparison_linear.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 5a. Comparison Plot - Linear Scale (Wide)
    fig = plt.figure(figsize=(15, 6))
    ax = plt.gca()
    
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Training Set', color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Validation Set', color=val_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Training and Validation Distributions (Linear Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
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
             fontsize=18)
    
    plt.savefig(os.path.join(save_dir, 'comparison_linear_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'comparison_linear_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 6. Comparison Plot - Log Scale (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Training Set', color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Validation Set', color=val_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Training and Validation Distributions (Log Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
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
             fontsize=18)
    
    plt.savefig(os.path.join(save_dir, 'comparison_log_normal.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'comparison_log_normal.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    # 6. Comparison Plot - Log Scale (Normal)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    plt.hist(train_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Training Set', color=train_color,
             edgecolor='white', linewidth=0.5)
    plt.hist(val_eigenvalues, bins=bins_calc, alpha=0.6,
             label='Validation Set', color=val_color,
             edgecolor='white', linewidth=0.5)
    
    plt.title('Comparison of Training and Validation Distributions (Log Scale)',
             pad=20, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.xlim(left=0)
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
             fontsize=18)
    
    plt.savefig(os.path.join(save_dir, 'comparison_log_wide.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(os.path.join(save_dir, 'comparison_log_wide.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()























if __name__ == "__main__":
    time.sleep(SLEEP_GLOB)
    data_dir = INP_DIR
    # check_graph_transformation(data_dir)
    use_z_coord = USE_Z_COORD_GLOB
    use_rotations = USE_ROT_GLOB
    use_gp_forces = False
    use_axial_stress = USE_AXIAL_STRESS_GLOB
    use_mode_shapes_as_features = False
    use_super_node = USE_SUPER_NODE_GLOB
    prediction_type = PREDICTION_TYPE_GLOB  # can be "buckling", "static_stress", "static_disp" or "mode_shape"
    virtual_edges_file= None

    data_dir = Path(data_dir)

    # TRAIN DATASET PROCESSING
    dataset_cache_file_check = os.path.join(data_dir, 'Train', f'dataset_cache_{prediction_type if not "static" in prediction_type else "static"}.pkl')
    normalizer_cache_file = os.path.join(data_dir, f"normalizer_cache.pkl")

    # CASE 1: TRAIN CACHE EXISTS AND NORMALIZER CACHE EXISTS
    if os.path.exists(dataset_cache_file_check) and os.path.exists(normalizer_cache_file):
        
        print("Loading cached normalizer...")
        with open(normalizer_cache_file, "rb") as f:
            normalizer = pickle.load(f)  

        train_data = load_folder_dataset(
            data_dir/'Train', 
            normalizer = normalizer,
            use_z_coord=use_z_coord,
            use_rotations=use_rotations,
            use_gp_forces=use_gp_forces,
            use_axial_stress=use_axial_stress,
            use_mode_shapes_as_features=use_mode_shapes_as_features,
            virtual_edges_file=virtual_edges_file,
            use_super_node=use_super_node,
            prediction_type=prediction_type
        )
    
    # CASE 2: TRAIN CACHE EXISTS AND NORMALIZER CACHE DOES NOT EXIST
    elif os.path.exists(dataset_cache_file_check) and not os.path.exists(normalizer_cache_file):
        train_data, normalizer = load_folder_dataset(
            data_dir/'Train', 
            normalizer = None,
            use_z_coord=use_z_coord,
            use_rotations=use_rotations,
            use_gp_forces=use_gp_forces,
            use_axial_stress=use_axial_stress,
            use_mode_shapes_as_features=use_mode_shapes_as_features,
            virtual_edges_file=virtual_edges_file,
            use_super_node=use_super_node,
            prediction_type=prediction_type
        )

        print("Caching normalizer...")
        with open(normalizer_cache_file, "wb") as f:
            pickle.dump(normalizer, f)
    else:
        train_data, normalizer = load_folder_dataset(
            data_dir/'Train', 
            normalizer = None,
            use_z_coord=use_z_coord,
            use_rotations=use_rotations,
            use_gp_forces=use_gp_forces,
            use_axial_stress=use_axial_stress,
            use_mode_shapes_as_features=use_mode_shapes_as_features,
            virtual_edges_file=virtual_edges_file,
            use_super_node=use_super_node,
            prediction_type=prediction_type
        )
        print("Caching normalizer...")
        with open(normalizer_cache_file, "wb") as f:
            pickle.dump(normalizer, f)
    
    if len(train_data) == 0:
        raise ValueError("No valid data was loaded for train. Please check input files.")
    
    # VALIDATION DATASET PROCESSING
    dataset_cache_file_check = os.path.join(data_dir, 'Val', f'dataset_cache_{prediction_type if not "static" in prediction_type else "static"}.pkl',)
        
    # CASE 1: VAL CACHE EXISTS
    if os.path.exists(dataset_cache_file_check):
        val_data = load_folder_dataset(
            data_dir/'Val', 
            normalizer = normalizer,
            use_z_coord=use_z_coord,
            use_rotations=use_rotations,
            use_gp_forces=use_gp_forces,
            use_axial_stress=use_axial_stress,
            use_mode_shapes_as_features=use_mode_shapes_as_features,
            virtual_edges_file=virtual_edges_file,
            use_super_node=use_super_node,
            prediction_type=prediction_type
        )
    # CASE 1: VAL CACHE NOT EXISTS
    else:
        val_data = load_folder_dataset(
            data_dir/'Val', 
            normalizer = normalizer,
            use_z_coord=use_z_coord,
            use_rotations=use_rotations,
            use_gp_forces=use_gp_forces,
            use_axial_stress=use_axial_stress,
            use_mode_shapes_as_features=use_mode_shapes_as_features,
            virtual_edges_file=virtual_edges_file,
            use_super_node=use_super_node,
            prediction_type=prediction_type
        )
    
    if len(val_data) == 0:
        raise ValueError("No valid data was loaded for validation. Please check input files.")
    
    # if "static" in prediction_type and use_super_node:
    #     for data in train_data:
    #         data.x = torch.cat([data.x[:, :6], data.x[:, -1:]], dim=1)
    #     for data in val_data:
    #         data.x = torch.cat([data.x[:, :6], data.x[:, -1:]], dim=1)
    # elif "static" in prediction_type and not use_super_node:
    #     for data in train_data:
    #         data.x = data.x[:, :6]
    #     for data in val_data:
    #         data.x = data.x[:, :6]
    
    if prediction_type == "static_stress":
        disp_dim = train_data[0].y.shape[1] - 3
        for data in train_data:
            data.y = data.y[:, disp_dim:]
        for data in val_data:
            data.y = data.y[:, disp_dim:]
    elif prediction_type == "static_disp":
        disp_dim = train_data[0].y.shape[1] - 3
        for data in train_data:                      
            data.y = data.y[:, :disp_dim]
        for data in val_data:
            data.y = data.y[:, :disp_dim]

########
    # Generate visualizations
    if PLOT_DATABASE_INFO_GLOB == True:
        vis_dir = os.path.join(OUTPUT_DIR, 'norm_split_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        train_eigenvalues = np.array([data.y.item() for data in train_data])
        val_eigenvalues = np.array([data.y.item() for data in val_data])
        visualize_split_distributions(train_eigenvalues, val_eigenvalues, vis_dir)

        vis_dir = os.path.join(OUTPUT_DIR, 'denorm_split_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        train_eigenvalues = normalizer.denormalize_eigenvalue(torch.tensor([data.y.reshape(-1)[0].item() if data.y.dim() > 0 else data.y.item() for data in train_data])).numpy()
        val_eigenvalues = normalizer.denormalize_eigenvalue(torch.tensor([data.y.reshape(-1)[0].item() if data.y.dim() > 0 else data.y.item() for data in val_data])).numpy()
        visualize_split_distributions2(train_eigenvalues, val_eigenvalues, vis_dir)
        sys.exit(1)
#########
    
    train_loader = DataLoader(train_data, BATCH_SIZE_GLOB, shuffle=True)
    val_loader = DataLoader(val_data, BATCH_SIZE_GLOB)
    
    num_node_features = train_data[0].num_node_features
    num_edge_features = train_data[0].num_edge_features
    
    # Get appropriate values for normalization based on prediction type
    if prediction_type == "buckling":
        allValues = [data.y.item() for data in train_data]
    elif prediction_type == "static_disp":
        allValues = [data.y for data in train_data]
    elif prediction_type == "static_stress":
        allValues = [data.y for data in train_data]
    elif prediction_type == "mode_shape":
        allValues = [data.y for data in train_data]

    mode = MODE_GLOB

    if mode == "auto":
        best_config = hyperparameter_optimization(train_loader, val_loader, num_node_features, num_edge_features, prediction_type, normalizer)
        print("Best trial config:", best_config) 
        SelectedLossFunction = best_config["loss_function"]
        model = train_gnn(best_config, data_loaders=(num_node_features, num_edge_features, train_loader, val_loader, allValues, prediction_type,normalizer))
    elif mode == "manual":
        config = CONFIG_MANUAL_GLOB
        SelectedLossFunction = config["loss_function"]
        model = train_gnn(config, data_loaders=(num_node_features, num_edge_features, train_loader, val_loader, allValues, prediction_type,normalizer))
    else:
        print("Invalid mode. Please choose 'auto' or 'manual'.")
        sys.exit(1)

    # Save model
    model_save_path = os.path.join(OUTPUT_DIR, f'ea_gnn_model_{prediction_type}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalizer': normalizer,
        'config': {
            'num_node_features': num_node_features,
            'num_edge_features': num_edge_features,
            'hidden_channels': config["hidden_channels"],
            'num_layers': config["num_layers"],
            'use_edge_attr': config["use_edge_attr"],
            'use_z_coord': use_z_coord,
            'use_rotations': use_rotations,
            'prediction_type': prediction_type,
            'pooling_layer': config["pooling_layer"],
            'dropout_rate': config["dropout_rate"],
            'model_name': config["model_name"]
        }
    }, model_save_path)
