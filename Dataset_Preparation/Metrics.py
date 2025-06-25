import torch
import numpy as np

def MAPE_error(predictions, targets, prediction_type="buckling", normalizer=None, threshold = 0.1):
    if prediction_type == "buckling":
        # Denormalize predictions and targets before calculating MAPE
        if normalizer is not None:
            predictions_denorm = normalizer.denormalize_eigenvalue(predictions)
            targets_denorm = normalizer.denormalize_eigenvalue(targets)
            return torch.mean(torch.abs((targets_denorm - predictions_denorm) / targets_denorm)) * 100
        else:
            return torch.mean(torch.abs((targets - predictions) / targets)) * 100
    elif prediction_type == "static_disp":
        high_disp_mask = torch.abs(targets) >= threshold
        return torch.mean(torch.abs((targets[high_disp_mask] - predictions[high_disp_mask]) / (targets[high_disp_mask] + 1e-8))) * 100
    elif prediction_type == "static_stress":
        high_stress_mask = torch.abs(targets) >= threshold
        return torch.mean(torch.abs((targets[high_stress_mask] - predictions[high_stress_mask]) / (targets[high_stress_mask] + 1e-8))) * 100
    elif prediction_type == "mode_shape":
        # For mode shapes, calculate MAPE on the normalized vectors
        pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
        target_norm = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-8)
        return torch.mean(torch.abs(pred_norm - target_norm)) * 100
    
def stress_errors(predictions, targets, batch=None, prediction_type="static_stress", threshold=0.1):
    """Calculate error metrics per graph and average them.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        batch: Batch indices for each node
        prediction_type: Type of prediction ("static_stress" or "static_disp")
        normalizer: Normalizer object for denormalization if needed
        threshold: Threshold for high/low regions (used for both stress and displacement)
    
    Returns:
        Dictionary containing various error metrics
    """
    if batch is None:
        # If no batch provided, treat as single graph
        batch = torch.zeros(len(predictions), dtype=torch.long, device=predictions.device)
    
    num_graphs = batch.max().item() + 1
    error_dict = {}
    
    # Initialize lists to store per-graph metrics
    if prediction_type == "static_stress":
        graph_metrics = {
            'max_x_val': [], 'max_x_mae': [], 'max_x_rel': [],
            'max_y_val': [], 'max_y_mae': [], 'max_y_rel': [],
            'max_xy_val': [], 'max_xy_mae': [], 'max_xy_rel': [],
            'mape_high': [], 're_high': [], 'rmse_high': [], 'mae_high': [], 'p90_high': [],
            'mape_low': [], 're_low': [], 'rmse_low': [], 'mae_low': [], 'p90_low': [],
            'mape': [], 're': [], 'rmse': [], 'mae': [], 'mse': [], 'p90': [],
            'max_mae': [], 'std_mae': [], 'p90_abs': []
        }
        
        # Process each graph separately
        for g in range(num_graphs):
            mask = batch == g
            graph_pred = predictions[mask]
            graph_target = targets[mask]
            
            abs_diff = torch.abs(graph_target - graph_pred)
            rel_diff = abs_diff / (torch.abs(graph_target) + 1e-8)
            
            # Maximum values for each stress component
            stress_components = ['x', 'y', 'xy']
            for i, comp in enumerate(stress_components):
                max_idx = torch.argmax(torch.abs(graph_target[:, i]))
                graph_metrics[f'max_{comp}_val'].append(np.abs(graph_target[max_idx, i].item()))
                graph_metrics[f'max_{comp}_mae'].append(abs_diff[max_idx, i].item())
                graph_metrics[f'max_{comp}_rel'].append((abs_diff[max_idx, i] / (torch.abs(graph_target[max_idx, i]) + 1e-8) * 100).item())
            
            # High stress regions
            high_stress_mask = torch.abs(graph_target) >= threshold
            if high_stress_mask.any():
                graph_metrics['mape_high'].append(torch.mean(rel_diff[high_stress_mask]).item() * 100)
                graph_metrics['re_high'].append((torch.linalg.vector_norm(abs_diff[high_stress_mask], ord=1) / 
                                              torch.linalg.vector_norm(graph_target[high_stress_mask], ord=1) * 100).item())
                graph_metrics['rmse_high'].append(torch.sqrt((graph_target[high_stress_mask]**2 - 
                                                            graph_pred[high_stress_mask]**2).mean()).item())
                graph_metrics['mae_high'].append(abs_diff[high_stress_mask].mean().item())
                graph_metrics['p90_high'].append(torch.quantile(rel_diff[high_stress_mask], 0.9).item() * 100)
            
            # Low stress regions
            low_stress_mask = torch.abs(graph_target) < threshold
            if low_stress_mask.any():
                graph_metrics['mape_low'].append(torch.mean(rel_diff[low_stress_mask]).item() * 100)
                graph_metrics['re_low'].append((torch.linalg.vector_norm(abs_diff[low_stress_mask], ord=1) / 
                                             torch.linalg.vector_norm(graph_target[low_stress_mask], ord=1) * 100).item())
                graph_metrics['rmse_low'].append(torch.sqrt((graph_target[low_stress_mask]**2 - 
                                                           graph_pred[low_stress_mask]**2).mean()).item())
                graph_metrics['mae_low'].append(abs_diff[low_stress_mask].mean().item())
                graph_metrics['p90_low'].append(torch.quantile(rel_diff[low_stress_mask], 0.9).item() * 100)
            
            # Overall metrics
            graph_metrics['mape'].append(torch.mean(rel_diff).item() * 100)
            graph_metrics['re'].append((torch.linalg.vector_norm(abs_diff, ord=1) / 
                                     torch.linalg.vector_norm(graph_target, ord=1) * 100).item())
            graph_metrics['rmse'].append(torch.sqrt((graph_target**2 - graph_pred**2).mean()).item())
            graph_metrics['mae'].append(abs_diff.mean().item())
            graph_metrics['mse'].append((graph_target**2 - graph_pred**2).mean().item())
            graph_metrics['p90'].append(torch.quantile(rel_diff, 0.9).item() * 100)
            graph_metrics['max_mae'].append(abs_diff.max().item())
            graph_metrics['std_mae'].append(abs_diff.std().item())
            graph_metrics['p90_abs'].append(torch.quantile(abs_diff, 0.9).item())
        
        # Calculate means across all graphs
        error_dict = {key: sum(values) for key, values in graph_metrics.items()}
        
    elif prediction_type == "static_disp":
        graph_metrics = {
            'max_disp_val': [], 'max_disp_mae': [], 'max_disp_rel': [],
            'max_x_val': [], 'max_x_mae': [], 'max_x_rel': [],
            'max_y_val': [], 'max_y_mae': [], 'max_y_rel': [],
            'mape_high': [], 're_high': [], 'rmse_high': [], 'mae_high': [], 'p90_high': [],
            'mape_low': [], 're_low': [], 'rmse_low': [], 'mae_low': [], 'p90_low': [],
            'mape': [], 're': [], 'rmse': [], 'mae': [], 'mse': [], 'p90': [],
            'max_mae': [], 'std_mae': [], 'p90_abs': []
        }
        
        # Process each graph separately
        for g in range(num_graphs):
            mask = batch == g
            graph_pred = predictions[mask]
            graph_target = targets[mask]
            
            abs_diff = torch.abs(graph_target - graph_pred)
            rel_diff = abs_diff / (torch.abs(graph_target) + 1e-8)
            
            # Calculate displacement magnitudes for thresholding
            target_magnitudes = torch.norm(graph_target, dim=1)
            
            # Maximum resultant displacement
            max_disp_idx = torch.argmax(target_magnitudes)
            max_loc_abs_error = torch.norm(abs_diff[max_disp_idx])
            
            graph_metrics['max_disp_val'].append(target_magnitudes[max_disp_idx].item())
            graph_metrics['max_disp_mae'].append(max_loc_abs_error.item())
            graph_metrics['max_disp_rel'].append((max_loc_abs_error / (target_magnitudes[max_disp_idx] + 1e-8) * 100).item())
            
            # Component-wise maximum analysis
            for i, comp in enumerate(['x', 'y']):
                max_comp_idx = torch.argmax(torch.abs(graph_target[:, i]))
                graph_metrics[f'max_{comp}_val'].append(np.abs(graph_target[max_comp_idx, i].item()))
                graph_metrics[f'max_{comp}_mae'].append(abs_diff[max_comp_idx, i].item())
                graph_metrics[f'max_{comp}_rel'].append((abs_diff[max_comp_idx, i] / 
                                                       (torch.abs(graph_target[max_comp_idx, i]) + 1e-8) * 100).item())
            
            # High displacement regions
            high_disp_mask = target_magnitudes >= threshold
            if high_disp_mask.any():
                graph_metrics['mape_high'].append(torch.mean(rel_diff[high_disp_mask]).item() * 100)
                graph_metrics['re_high'].append((torch.linalg.vector_norm(abs_diff[high_disp_mask], ord=1) / 
                                              torch.linalg.vector_norm(graph_target[high_disp_mask], ord=1) * 100).item())
                graph_metrics['rmse_high'].append(torch.sqrt((graph_target[high_disp_mask]**2 - 
                                                            graph_pred[high_disp_mask]**2).mean()).item())
                graph_metrics['mae_high'].append(abs_diff[high_disp_mask].mean().item())
                graph_metrics['p90_high'].append(torch.quantile(rel_diff[high_disp_mask], 0.9).item() * 100)
            
            # Low displacement regions
            low_disp_mask = target_magnitudes < threshold
            if low_disp_mask.any():
                graph_metrics['mape_low'].append(torch.mean(rel_diff[low_disp_mask]).item() * 100)
                graph_metrics['re_low'].append((torch.linalg.vector_norm(abs_diff[low_disp_mask], ord=1) / 
                                             torch.linalg.vector_norm(graph_target[low_disp_mask], ord=1) * 100).item())
                graph_metrics['rmse_low'].append(torch.sqrt((graph_target[low_disp_mask]**2 - 
                                                           graph_pred[low_disp_mask]**2).mean()).item())
                graph_metrics['mae_low'].append(abs_diff[low_disp_mask].mean().item())
                graph_metrics['p90_low'].append(torch.quantile(rel_diff[low_disp_mask], 0.9).item() * 100)
            
            # Overall metrics
            graph_metrics['mape'].append(torch.mean(rel_diff).item() * 100)
            graph_metrics['re'].append((torch.linalg.vector_norm(abs_diff, ord=1) / 
                                     torch.linalg.vector_norm(graph_target, ord=1) * 100).item())
            graph_metrics['rmse'].append(torch.sqrt((graph_target**2 - graph_pred**2).mean()).item())
            graph_metrics['mae'].append(abs_diff.mean().item())
            graph_metrics['mse'].append((graph_target**2 - graph_pred**2).mean().item())
            graph_metrics['p90'].append(torch.quantile(rel_diff, 0.9).item() * 100)
            graph_metrics['max_mae'].append(abs_diff.max().item())
            graph_metrics['std_mae'].append(abs_diff.std().item())
            graph_metrics['p90_abs'].append(torch.quantile(abs_diff, 0.9).item())
        
        # Calculate means across all graphs
        error_dict = {key: sum(values) for key, values in graph_metrics.items()}
    
    else:
        raise NotImplementedError(f"Error metrics not implemented for prediction type: {prediction_type}")

    return error_dict