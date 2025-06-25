import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_loss_function(loss_name, allValues=None, use_z_coord=False, use_rotations=False):
    if loss_name == "static_mixed":
        return StaticAnalysisLoss(alpha=0.1)
    
    elif loss_name == "static_mse":
        return StaticAnalysisLoss(alpha=0.0)
    
    elif loss_name == "static_relative":
        return StaticAnalysisLoss(alpha=1.0)
    elif loss_name == "static_stress":
        return StaticFocalStressLoss()
    elif loss_name == "static_mae":
        return StaticAnalysisLoss_MAE()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "relative_error":
        return RelativeErrorLoss()
    elif loss_name == "log_cosh":
        return LogCoshLoss()
    elif loss_name == "eigenvalue":
        return EigenvalueLoss(alpha=0.5, beta=0.5)
    elif loss_name == "order_preserving":
        return OrderPreservingLoss()
    elif loss_name == "focal":
        return FocalLossRegression(allValues, alpha=1.0, gamma=2.0, num_bins=100, device=device)
    elif loss_name == "mape":
        return MAPE()
    elif loss_name == "graph_mse":
        return GraphMSELoss()
    elif loss_name == "graph_mae":
        return GraphMAELoss()
    elif loss_name == "graph_rel":
        return GraphRelativeError()
    elif loss_name == "graph_mixed":
        return GraphMixedError()
    elif loss_name == "graph_max_rel":
        return GraphMaxComponentRelativeError()
    elif loss_name == "graph_rel_scaled":
        return ScaledGraphRELoss()
    elif loss_name == "graph_mae_scaled":
        return ScaledGraphMAELoss()
    elif loss_name == "graph_mse_scaled":
        return ScaledGraphMSELoss()
    elif loss_name == "mae":
        return MAE()
    elif loss_name == "rse":
        return RSE(allValues)
    elif loss_name == "rrse":
        return RRSE()
    elif loss_name == "rrse1":
        return RRSE1()
    elif loss_name == "msle":
        return MSLELoss()
    elif loss_name == "focal_rrse":
        return FocalRRSE(allValues, alpha=1.0, gamma=2.0, num_bins=100, penalty_factor=10, device=device)
    elif loss_name == "focal_mape":
        return FocalMAPE(allValues, alpha=1.0, gamma=2.0, num_bins=100, device=device)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")



class RegulatedStressLoss(nn.Module):
    def __init__(self, threshold=0.3, base_scale=1.0, count_scale=0.01):
        super().__init__()
        self.threshold = threshold
        self.base_scale = base_scale
        self.count_scale = count_scale
        
    def msle(self, pred, target):
        # Modified MSLE to handle potential negative values
        return torch.square(torch.log1p(torch.abs(pred)) - torch.log1p(torch.abs(target)))
    
    def forward(self, pred, target, pprint=False):
        pred_abs = torch.abs(pred)
        target_abs = torch.abs(target)
        
        # Create masks
        high_stress_mask = target_abs >= self.threshold
        wrong_pred_mask = (target_abs < self.threshold) & (pred_abs >= self.threshold)
        
        # Calculate number of elements in each case
        n_wrong = wrong_pred_mask.sum().item()
        n_high = high_stress_mask.sum().item()
        
        # Calculate MSE losses for both cases
        wrong_loss = torch.square(pred_abs[wrong_pred_mask] - target_abs[wrong_pred_mask])
        high_loss = torch.square(pred_abs[high_stress_mask] - target_abs[high_stress_mask])
        
        # Calculate mean losses
        if n_wrong > 0:
            mean_wrong = wrong_loss.mean()
            # Scale wrong prediction loss based on number of cases
            count_factor = 1.0 + self.count_scale * n_wrong
            wrong_final = mean_wrong * count_factor
        else:
            wrong_final = torch.tensor(0.0, device=pred.device)
            
        if n_high > 0:
            mean_high = high_loss.mean()
        else:
            mean_high = torch.tensor(0.0, device=pred.device)
            
        # Balance the scales of the losses
        if n_wrong > 0 and n_high > 0:
            # Dynamically scale high stress loss to be similar magnitude
            scale_factor = (wrong_final / (mean_high + 1e-6)).detach()
            scale_factor = torch.clamp(scale_factor, 0.1, 10.0)  # Prevent extreme scaling
            high_final = mean_high * scale_factor * self.base_scale
        else:
            high_final = mean_high * self.base_scale
            
        # Combine losses
        total_loss = wrong_final + high_final

        if pprint:
            print(
                f"stress<0.1&pred>0.1: #{n_wrong}, "
                f"small error: {wrong_final.item():.5f}\n"
                f"stress>0.1: #{n_high}, "
                f"big error: {high_final.item():.5f}\n"
                f"stress>0.1 mean of preds: {pred_abs[high_stress_mask].mean().item():.5f}\n"
                f"stress>0.1 mean of target: {target_abs[high_stress_mask].mean().item():.5f}\n"
                f"total loss: {total_loss.item():.1f}"
            )
            
        return total_loss

class StaticAnalysisLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(StaticAnalysisLoss, self).__init__()
        self.alpha = alpha  # Balance between relative and absolute error
            
    def forward(self, pred, target):
        # Calculate relative error
        rel_loss = torch.mean(torch.abs((pred - target) / (target + 1e-8)))
        
        # Calculate MSE
        mse_loss = F.mse_loss(pred, target)
        
        # Combine relative and MSE losses with alpha parameter
        total_loss = self.alpha * rel_loss + (1 - self.alpha) * mse_loss
        return total_loss
    
class StaticAnalysisLoss_MAE(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, pred, target):
        # # Split predictions and targets
        # mae= torch.mean(torch.abs(pred - target))
        # mse_loss = F.mse_loss(pred, target)
        
        error_norm = torch.norm(pred - target, p=1) 
        # Calculate L1 norm of true values
        # target_norm = torch.norm(target, p=1) + self.epsilon
        
        return error_norm 
        # return mae + mse_loss

class MSLELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MSLELoss, self).__init__()
        self.epsilon = epsilon  # Small constant to prevent log(0)
        self.threshold = 0.7
    def forward(self, pred, target, pprint=False):
        # Ensure non-negative predictions and targets
        pred = torch.clamp(pred, min=0)  # Clamp predictions to be >= 0
        target = torch.clamp(target, min=0)
        
        # Compute the logarithm of (1 + prediction) and (1 + target)
        log_pred = torch.log1p(pred + self.epsilon)
        log_target = torch.log1p(target + self.epsilon)
        
        # Mean squared error between the logarithmic values
        loss = ((log_pred - log_target) ** 2)
        mean_loss = loss.mean() 

        if pprint:
            pred_abs = torch.abs(pred)
            target_abs = torch.abs(target)
            
            # Create masks for different regions
            high_stress_mask = target_abs >= self.threshold
            wrong_pred_mask = (target_abs < self.threshold) & (pred_abs >= self.threshold)
        
            print(
                f"stress<0.1&pred>0.1: #{wrong_pred_mask.sum().item()}, "
                f"small error: {loss[wrong_pred_mask].mean().item():.5f}\n"
                f"stress>0.1: #{high_stress_mask.sum().item()}, "
                f"big error: {loss[high_stress_mask].mean().item():.5f}\n"
                f"stress>0.1 mean of preds: {pred_abs[high_stress_mask].mean().item():.5f}\n"
                f"stress>0.1 mean of target: {target_abs[high_stress_mask].mean().item():.5f}\n"
                f"total loss: {mean_loss.item():.1f}"
            )
        return mean_loss

class StaticFocalStressLoss(nn.Module):
    def __init__(self,  alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = 0.7
    
    def forward(self, y_pred, y_true, mask, pprint=False):
        mask = mask.repeat(1,3)
        # 1. MSE Loss bileşeni
        mse_loss = torch.mean((y_true[mask] - y_pred[mask])**2)
        
        # 2. Focal Loss benzeri ağırlıklandırma
        # Büyük hataları daha çok önemse
        error = torch.abs(y_true[mask] - y_pred[mask])
        focal_weights = (error) ** self.gamma
        focal_loss = torch.mean(focal_weights * error)
        
        # 3. Stress büyüklüğüne göre ağırlıklandırma
        stress_weights = torch.abs(y_true[mask]) + 1  # Küçük değerler için ~1, büyük değerler için daha yüksek
        weighted_loss = torch.mean(stress_weights * error)
        
        # Kombinasyon
        total_loss = mse_loss + self.alpha * (focal_loss + weighted_loss)
            
        
        if pprint:
            target_abs = torch.abs(y_true)
            pred_abs = torch.abs(y_pred)
            high_stress_mask = target_abs >= self.threshold
            low_pred_mask = (target_abs < self.threshold)
            print(
                f"stress<0.1 mean of preds: {pred_abs[low_pred_mask].mean().item():.5f}\n"
                f"stress<0.1 mean of target: {target_abs[low_pred_mask].mean().item():.5f}\n"
                f"stress>0.1 mean of preds: {pred_abs[high_stress_mask].mean().item():.5f}\n"
                f"stress>0.1 mean of target: {target_abs[high_stress_mask].mean().item():.5f}\n"
                f"total loss: {total_loss.item():.1f}"
            )
        return total_loss

class Exponential(nn.Module):
    def __init__(self, threshold=0.1, small_stress_weight=0.1, exp_factor=1.2, overestimation_penalty=2.0):
        super().__init__()
        self.threshold = threshold
        self.small_stress_weight = small_stress_weight
        self.exp_factor = exp_factor
        self.overestimation_penalty = overestimation_penalty
    
    def forward(self, pred, target, pprint=False):
        pred_abs = torch.abs(pred)
        target_abs = torch.abs(target)
        
        # Create masks for different regions
        high_stress_mask = target_abs >= self.threshold
        wrong_pred_mask = (target_abs < self.threshold) & (pred_abs >= self.threshold)
        
        # Basic squared error
        base_loss = torch.square(pred_abs - target_abs)
        
        # Progressive overestimation penalty
        overestimation_ratio = pred_abs / (target_abs + 1e-6)  # Add small epsilon to prevent division by zero
        overestimation_mask = overestimation_ratio > 1.2  # Over 20% higher than target
        
        # Penalty increases quadratically with overestimation ratio
        overestimation_loss = torch.where(
            overestimation_mask,
            self.overestimation_penalty * torch.square(overestimation_ratio - 1.0) * base_loss,
            torch.zeros_like(pred_abs)
        )
        
        # More gentle weight factor for genuine high stresses
        weight = torch.where(
            target_abs >= self.threshold,
            1.0 + torch.pow(target_abs, self.exp_factor),  # Base weight of 1.0 plus exponential component
            self.small_stress_weight * torch.ones_like(target_abs)
        )
        
        # Calculate losses for different regions
        high_stress_loss = (base_loss[high_stress_mask] + overestimation_loss[high_stress_mask]) * weight[high_stress_mask]
        low_stress_loss = base_loss[wrong_pred_mask] * self.small_stress_weight
        
        # Total loss
        total_loss = torch.cat([high_stress_loss, low_stress_loss])
        mean_loss = total_loss.mean()
        
        if pprint:
            print(
                f"stress<{0.1}&pred>{0.1}: #{wrong_pred_mask.sum().item()}, "
                f"small error: {low_stress_loss.mean().item():.5f}\n"
                f"stress>{0.1}: #{high_stress_mask.sum().item()}, "
                f"big error: {high_stress_loss.mean().item():.5f}\n"
                f"stress>{0.1} mean of preds: {pred_abs[high_stress_mask].mean().item():.5f}\n"
                f"stress>{0.1} mean of target: {target_abs[high_stress_mask].mean().item():.5f}\n"
                f"total loss: {mean_loss.item():.1f}"
            )
            
        return mean_loss

class GraphMaxComponentRelativeError(nn.Module):
    """
    Implements relative error loss at maximum target value locations.
    For each graph and each component:
    1. Finds the location of maximum absolute target value
    2. Computes the relative error at that location
    3. Averages these errors across components and graphs
    
    Args:
        epsilon (float): Small constant to prevent division by zero
    """
    def __init__(self, epsilon=1e-8):
        super(GraphMaxComponentRelativeError, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target, batch, x):
        # If no batch index provided, treat as single graph
        if batch is None:
            # For each component, find location of max target value
            target_abs = torch.abs(target)
            max_indices = torch.argmax(target_abs, dim=0)
            
            # Get predictions and targets at max locations
            max_target = torch.gather(target, 0, max_indices.unsqueeze(0)).squeeze(0)
            max_pred = torch.gather(pred, 0, max_indices.unsqueeze(0)).squeeze(0)
            
            # Compute relative errors at max locations
            rel_errors = torch.abs(max_pred - max_target) / (torch.abs(max_target) + self.epsilon)
            
            # Return mean across components
            return torch.mean(rel_errors)
            
        # Get number of graphs in batch
        num_graphs = batch.max().item() + 1
        num_components = pred.shape[1] if len(pred.shape) > 1 else 1
        
        # Initialize tensor to store per-graph relative errors at max locations
        graph_rel_errors = torch.zeros((num_graphs, num_components), device=pred.device)
        
        # Process each graph separately
        for i in range(num_graphs):
            graph_mask = (batch == i)
            graph_pred = pred[graph_mask]
            graph_target = target[graph_mask]
            
            # For each component, find location of max target value
            target_abs = torch.abs(graph_target)
            max_indices = torch.argmax(target_abs, dim=0)
            
            # Get predictions and targets at max locations
            max_target = torch.gather(graph_target, 0, max_indices.unsqueeze(0)).squeeze(0)
            max_pred = torch.gather(graph_pred, 0, max_indices.unsqueeze(0)).squeeze(0)
            
            # Compute relative errors at max locations
            graph_rel_errors[i] = torch.abs(max_pred - max_target) / (torch.abs(max_target) + self.epsilon)
        
        # Return mean of relative errors at max locations across all graphs
        return torch.mean(graph_rel_errors)*10000
    
class GraphRelativeError(nn.Module):
    """
    Implements graph-wise P90 (90th percentile) relative error loss.
    For each graph in the batch, computes P90 of relative errors separately before averaging.
    
    Args:
        epsilon (float): Small constant to prevent division by zero
        percentile (float): The percentile to compute (default 0.9 for P90)
    """
    def __init__(self, epsilon=0.1):
        super(GraphRelativeError, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target, batch, x):
        # If no batch index provided, compute over all nodes
        if batch is None:
            rel_errors = torch.abs(pred - target) / (torch.abs(target) + self.epsilon)
            return torch.mean(rel_errors)
            
        # Get number of graphs in batch
        num_graphs = batch.max().item() + 1
        
        # Initialize tensor to store per-graph P90 values
        graph_p90s = torch.zeros(num_graphs, device=pred.device)
        
        # Compute P90 for each graph separately
        for i in range(num_graphs):
            graph_mask = (batch == i)
            graph_pred = pred[graph_mask]
            graph_target = target[graph_mask]
            
            # Compute relative errors for this graph
            rel_errors = torch.abs(graph_pred - graph_target) / (torch.abs(graph_target) + self.epsilon)
            
            # Compute P90 for this graph
            graph_p90s[i] = torch.mean(rel_errors)
            # graph_p90s[i] = torch.quantile(rel_errors, self.percentile)
        
        # Return mean of per-graph P90s
        return torch.mean(graph_p90s)*10000
    
class GraphMixedError(nn.Module):
    """
    Implements graph-wise P90 (90th percentile) relative error loss.
    For each graph in the batch, computes P90 of relative errors separately before averaging.
    
    Args:
        epsilon (float): Small constant to prevent division by zero
        percentile (float): The percentile to compute (default 0.9 for P90)
    """
    def __init__(self, epsilon=1e-8, percentile=0.2):
        super(GraphMixedError, self).__init__()
        self.epsilon = epsilon
        self.percentile = percentile
    
    def forward(self, pred, target, batch, x):
        # If no batch index provided, compute over all nodes
        if batch is None:
            rel_errors = torch.abs(pred - target) / (torch.abs(target) + self.epsilon)
            return 0.2*torch.quantile(rel_errors, self.percentile) + 0.8*torch.mean(torch.abs(pred - target))
            
        # Get number of graphs in batch
        num_graphs = batch.max().item() + 1
        
        # Initialize tensor to store per-graph P90 values
        graph_p90s = torch.zeros(num_graphs, device=pred.device)
        graph_losses = torch.zeros(num_graphs, device=pred.device)
        
        # Compute P90 for each graph separately
        for i in range(num_graphs):
            graph_mask = (batch == i)
            graph_pred = pred[graph_mask]
            graph_target = target[graph_mask]
            
            # Compute relative errors for this graph
            rel_errors = torch.abs(graph_pred - graph_target) / (torch.abs(graph_target) + self.epsilon)
            
            # Compute P90 for this graph
            graph_p90s[i] = torch.quantile(rel_errors, self.percentile)
            graph_losses[i] = torch.mean(torch.abs(graph_pred - graph_target))
        # Return mean of per-graph P90s
        return 0.2*torch.mean(graph_p90s) + 0.8*torch.mean(graph_losses)
    
class GraphMSELoss(nn.Module):
    """
    Implements graph-wise Mean Absolute Error loss.
    For each graph in the batch, computes MAE separately before averaging.
    """
    def __init__(self, alpha=0.5):
        super(GraphMSELoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred, target, batch, x):
        # If no batch index provided, assume single graph
        if batch is None:
            return torch.mean(torch.abs(pred - target)**2)
            
        # Get number of graphs in batch
        num_graphs = batch.max().item() + 1
        
        # Initialize tensor to store per-graph losses
        graph_losses = torch.zeros(num_graphs, device=pred.device)
        
        # Compute MAE for each graph separately
        for i in range(num_graphs):
            graph_mask = (batch == i)
            graph_pred = pred[graph_mask]
            graph_target = target[graph_mask]
            
            # Compute MAE for this graph
            graph_losses[i] = torch.mean(torch.abs(graph_pred**2 - graph_target**2))
        
        # Return mean of per-graph losses
        return torch.mean(graph_losses)*10000
    
class GraphMAELoss(nn.Module):
    """
    Implements graph-wise Mean Absolute Error loss.
    For each graph in the batch, computes MAE separately before averaging.
    """
    def __init__(self, alpha=0.5):
        super(GraphMAELoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred, target, batch, x):
        # If no batch index provided, assume single graph
        if batch is None:
            return torch.mean(torch.abs(pred - target))
            
        # Get number of graphs in batch
        num_graphs = batch.max().item() + 1
        
        # Initialize tensor to store per-graph losses
        graph_losses = torch.zeros(num_graphs, device=pred.device)
        
        # Compute MAE for each graph separately
        for i in range(num_graphs):
            graph_mask = (batch == i)
            graph_pred = pred[graph_mask]
            graph_target = target[graph_mask]
            
            # Compute MAE for this graph
            graph_losses[i] = torch.mean(torch.abs(graph_pred - graph_target))
        
        # Return mean of per-graph losses
        return torch.mean(graph_losses)*10000
    
class ScaledGraphMAELoss(nn.Module):
    """
    Implements graph-wise Mean Absolute Error loss with force magnitude scaling.
    For each graph in the batch, computes MAE separately and scales up by total force magnitude before averaging.
    """
    def __init__(self, force_scaling=True, min_scale=0.1):
        super(ScaledGraphMAELoss, self).__init__()
        self.force_scaling = force_scaling
        self.min_scale = min_scale  # Minimum scaling factor
    
    def compute_total_force(self, x):
        """Compute total force magnitude for the given nodes"""
        # Assuming force features are at indices 3:5 for 2D or 3:6 for 3D
        force_features = x[:, 3:5]  # Adjust indices based on feature arrangement
        force_magnitudes = torch.norm(force_features, dim=1)  # Compute magnitude of force vectors
        return torch.sum(force_magnitudes)
    
    def forward(self, pred, target, batch, x):
        """
        Args:
            pred: Predicted values
            target: Target values
            batch: PyG batch object containing node features (x) and batch indices (batch.batch)
        """
        # If no batch indices provided, assume single graph
        if batch is None:
            if self.force_scaling:
                total_force = self.compute_total_force(x)
                scale_factor = max(total_force.item(), self.min_scale)
                return torch.mean(torch.abs(pred - target)) * scale_factor
            return torch.mean(torch.abs(pred - target))
        
        # Get number of graphs in batch
        num_graphs = batch.max().item() + 1
        
        # Initialize tensor to store per-graph losses
        graph_losses = torch.zeros(num_graphs, device=pred.device)
        
        # Compute scaled MAE for each graph separately
        for i in range(num_graphs):
            graph_mask = (batch == i)
            graph_pred = pred[graph_mask]
            graph_target = target[graph_mask]
            
            # Compute MAE for this graph
            graph_mae = torch.mean(torch.abs(graph_pred - graph_target))
            
            if self.force_scaling:
                # Get total force magnitude for this graph
                graph_forces = self.compute_total_force(x)
                scale_factor = max(graph_forces.item(), self.min_scale)
                # Scale up the loss for graphs with larger forces
                graph_mae = graph_mae * scale_factor
            
            graph_losses[i] = graph_mae
        
        # Return mean of per-graph scaled losses
        return torch.mean(graph_losses)*100
    
class ScaledGraphMSELoss(nn.Module):
    """
    Implements graph-wise Mean Absolute Error loss with force magnitude scaling.
    For each graph in the batch, computes MAE separately and scales up by total force magnitude before averaging.
    """
    def __init__(self, force_scaling=True, min_scale=0.1):
        super(ScaledGraphMSELoss, self).__init__()
        self.force_scaling = force_scaling
        self.min_scale = min_scale  # Minimum scaling factor
    
    def compute_total_force(self, x):
        """Compute total force magnitude for the given nodes"""
        # Assuming force features are at indices 3:5 for 2D or 3:6 for 3D
        force_features = x[:, 3:5]  # Adjust indices based on feature arrangement
        force_magnitudes = torch.norm(force_features, dim=1)  # Compute magnitude of force vectors
        return torch.sum(force_magnitudes)
    
    def forward(self, pred, target, batch, x):
        """
        Args:
            pred: Predicted values
            target: Target values
            batch: PyG batch object containing node features (x) and batch indices (batch.batch)
        """
        # If no batch indices provided, assume single graph
        if batch is None:
            if self.force_scaling:
                total_force = self.compute_total_force(x)
                scale_factor = max(total_force.item(), self.min_scale)
                return torch.mean(torch.abs(pred - target)) * scale_factor
            return torch.mean(torch.abs(pred - target))
        
        # Get number of graphs in batch
        num_graphs = batch.max().item() + 1
        
        # Initialize tensor to store per-graph losses
        graph_losses = torch.zeros(num_graphs, device=pred.device)
        
        # Compute scaled MAE for each graph separately
        for i in range(num_graphs):
            graph_mask = (batch == i)
            graph_pred = pred[graph_mask]
            graph_target = target[graph_mask]
            
            # Compute MAE for this graph
            graph_mse = torch.mean(torch.abs(graph_pred**2 - graph_target**2))
            
            if self.force_scaling:
                # Get total force magnitude for this graph
                graph_forces = self.compute_total_force(x)
                scale_factor = max(graph_forces.item(), self.min_scale)
                # Scale up the loss for graphs with larger forces
                graph_mse = graph_mse * scale_factor
            
            graph_losses[i] = graph_mse
        
        # Return mean of per-graph scaled losses
        return torch.mean(graph_losses)*100
    
class ScaledGraphRELoss(nn.Module):
    """
    Implements graph-wise Relative Error loss with force magnitude scaling.
    RE is calculated using vector norms as in Metrics.py
    """
    def __init__(self, force_scaling=True, min_scale=0.1):
        super(ScaledGraphRELoss, self).__init__()
        self.force_scaling = force_scaling
        self.min_scale = min_scale  # Minimum scaling factor
    
    def compute_total_force(self, x):
        """Compute total force magnitude for the given nodes"""
        # Assuming force features are at indices 3:5 for 2D or 3:6 for 3D
        force_features = x[:, 3:5]  # Adjust indices based on feature arrangement
        force_magnitudes = torch.norm(force_features, dim=1)  # Compute magnitude of force vectors
        return torch.sum(force_magnitudes)
    
    def compute_relative_error(self, pred, target):
        """
        Compute relative error using vector norms.
        Following the RE calculation pattern from Metrics.py
        """
        error_norm = torch.linalg.vector_norm(pred - target, ord=1)
        target_norm = torch.linalg.vector_norm(target, ord=1)
        
        # Add small epsilon to prevent division by zero
        return error_norm / (target_norm + 1e-8)
    
    def forward(self, pred, target, batch, x):
        """
        Args:
            pred: Predicted values
            target: Target values
            batch: PyG batch object containing node features (x) and batch indices (batch.batch)
        """
        # If no batch indices provided, assume single graph
        if batch is None:
            if self.force_scaling:
                total_force = self.compute_total_force(x)
                scale_factor = max(total_force.item(), self.min_scale)
                return self.compute_relative_error(pred, target) * scale_factor
            return self.compute_relative_error(pred, target)
        
        # Get number of graphs in batch
        num_graphs = batch.max().item() + 1
        
        # Initialize tensor to store per-graph losses
        graph_losses = torch.zeros(num_graphs, device=pred.device)
        
        # Compute scaled RE for each graph separately
        for i in range(num_graphs):
            graph_mask = (batch == i)
            graph_pred = pred[graph_mask]
            graph_target = target[graph_mask]
            
            # Compute RE for this graph
            graph_re = self.compute_relative_error(graph_pred, graph_target)
            
            if self.force_scaling:
                # Get total force magnitude for this graph
                graph_forces = self.compute_total_force(x)
                scale_factor = max(graph_forces.item(), self.min_scale)
                # Scale up the loss for graphs with larger forces
                graph_re = graph_re * scale_factor
            
            graph_losses[i] = graph_re
        
        # Return mean of per-graph scaled relative errors
        return torch.mean(graph_losses)*100
    
class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 0.1

    def forward(self, pred, target, pprint=False):
        loss = torch.mean(torch.abs(pred - target)**2)
        # loss1 = torch.linalg.vector_norm(pred - target,ord=2)
        if pprint:
            pred_abs = torch.abs(pred)
            target_abs = torch.abs(target)
            
            # Create masks for different regions
            high_stress_mask = target_abs >= self.threshold
            wrong_pred_mask = (target_abs < self.threshold) & (pred_abs >= self.threshold)
        
            print(
                f"stress<0.1&pred>0.1: #{wrong_pred_mask.sum().item()}, "
                f"small error: {torch.abs(pred - target)[wrong_pred_mask].mean().item():.5f}\n"
                f"stress>0.1: #{high_stress_mask.sum().item()}, "
                f"big error: {torch.abs(pred - target)[high_stress_mask].mean().item():.5f}\n"
                f"stress>0.1 mean of preds: {pred_abs[high_stress_mask].mean().item():.5f}\n"
                f"stress>0.1 mean of target: {target_abs[high_stress_mask].mean().item():.5f}\n"
                f"total loss: {loss.item():.1f}"
            )
        return loss
    
# class MAE(nn.Module):
#     def __init__(self, weight_factor=1.0):
#         super(MAE, self).__init__()
#         self.weight_factor = weight_factor
        
#     def forward(self, pred, target):
#         # Calculate absolute stress values for weighting
#         stress_magnitude = torch.abs(target)
        
#         # Create weights based on stress magnitude
#         # Higher stresses get higher weights
#         weights = self.weight_factor * stress_magnitude
        
#         # Calculate MSE with weights
#         squared_error = torch.abs(pred - target)
#         weighted_error = weights * squared_error
        
#         return torch.mean(weighted_error)


# Standalone RRSE loss for other uses
class RRSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        numerator = torch.mean((pred - target)**2)
        denominator = torch.mean(target**2)
        return torch.sqrt(numerator / (denominator + 1e-8))


class RelativeErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(RelativeErrorLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target) / (torch.abs(target) + self.epsilon))

class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.log(torch.cosh(pred - target)))

class EigenvalueLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(EigenvalueLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        rel_error = torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8))
        return self.alpha * mse_loss + self.beta * rel_error

class OrderPreservingLoss(nn.Module):
    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        order_loss = F.relu(-(pred[:, None] - pred[None, :]) * (target[:, None] - target[None, :])).mean()
        return mse_loss + order_loss

class FocalLossRegression(nn.Module):
    def __init__(self, values, alpha=1.0, gamma=2.0, num_bins=10, penalty_factor=2.0, device = 'cpu'):
        super(FocalLossRegression, self).__init__()
        self.alpha = alpha  # Scaling factor for the loss
        self.gamma = gamma  # Focusing parameter
        self.num_bins = num_bins  # Number of bins for the histogram
        self.penalty_factor = penalty_factor  # Factor to increase loss for out-of-bounds predictions

        # Convert allValues to a torch tensor
        values = torch.tensor(values, dtype=torch.float32)

        # Calculate bounds
        self.min_val, self.max_val = torch.min(values), torch.max(values)

        # Calculate the histogram of allValues
        hist, self.bin_edges = torch.histogram(values, bins=self.num_bins)

        # Calculate the frequency of each bin
        frequencies = hist.float() / len(values)

        # Identify the non-zero bins for interpolation
        zero_mask = hist == 0
        zero_indices = torch.where(zero_mask)[0]
        


        # Fill with left index value
        for idx in zero_indices:
            if idx==0:
                continue
            left_index = idx-1
            # right_index = nonzero_indices[nonzero_indices > i][0]
            left_freq = frequencies[left_index]
            # right_freq = interpolated_frequencies[right_index]
            frequencies[idx] = left_freq

        self.weights = 1.0 / (frequencies + 1)
        self.weights = self.weights/self.weights.sum()

        if num_bins>99:
            self.weights = self.smooth_weights(self.weights, 9)
        
        self.bin_edges = self.bin_edges.to(device)
        self.weights = self.weights.to(device)
        # from matplotlib import pyplot as plt
        # plt.plot(self.bin_edges[1:],  self.weights)
        # plt.show()


    def forward(self, predictions, targets):
        # Compute the absolute difference between predictions and targets
        errors = torch.abs(predictions - targets)

        # Find the bin index for each target value
        bin_indices = torch.bucketize(targets, self.bin_edges[1:], right=True) - 1  # Use `-1` to make index 0-based
        bin_indices[bin_indices<0] = 0
        bin_indices[bin_indices>self.num_bins-1] = self.num_bins-1
        # Assign weights based on bin indices
        target_weights = self.weights[bin_indices]

        # Apply penalty for out-of-bound predictions
        
        out_of_bounds_mask = (predictions < self.min_val) | (predictions > self.max_val)
        target_weights[out_of_bounds_mask] = self.penalty_factor


        # Calculate the focal loss
        loss = self.alpha * (target_weights * (errors ** self.gamma)).mean()

        return loss

    @staticmethod
    def smooth_weights(weights, smoothing_kernel_size):
        # Create a smoothing kernel (moving average)
        kernel = torch.ones(smoothing_kernel_size) / smoothing_kernel_size
        # Apply smoothing using 1D convolution
        weights_padded = F.pad(weights.unsqueeze(0).unsqueeze(0), (smoothing_kernel_size // 2, smoothing_kernel_size // 2), mode='reflect')
        smoothed_weights = F.conv1d(weights_padded, kernel.unsqueeze(0).unsqueeze(0)).squeeze()
        return smoothed_weights



# Mean Absolute Percentage Error (MAPE)
# class MAPE(nn.Module):
#     def __init__(self,):
#         super().__init__()

#     def forward(self, y_pred, y_true):
#         loss = (torch.abs(y_pred - y_true)/y_true)
#         return torch.mean(loss)

# class MAPE(nn.Module):
#     def __init__(self, epsilon=1e-8):
#         super().__init__()
#         self.epsilon = epsilon

#     def forward(self, y_pred, y_true):
#         loss = torch.abs((y_true - y_pred) / (torch.abs(y_true) + self.epsilon))
#         return torch.mean(loss) * 100  # Return percentage
class MAPE(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        loss = torch.abs((y_true - y_pred))
        return torch.mean(loss)  # Return percentage
# Relative Squared Error (RSE)
# class RSE(nn.Module):
#     def __init__(self, values):
#         super().__init__()
#         values = torch.tensor(values, dtype=torch.float32)
#         self.y_mean = values.mean()

#     def forward(self, y_pred, y_true):
#         loss = ((y_pred - y_true)**2/(y_true - self.y_mean)**2)
#         return torch.mean(loss)

class RSE(nn.Module):
    def __init__(self, values, epsilon=1e-8):
        super().__init__()
        values = torch.tensor(values, dtype=torch.float32)
        self.y_mean = values.mean()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        numerator = torch.mean((y_pred - y_true)**2)
        denominator = torch.mean((y_true - self.y_mean)**2) + self.epsilon
        return torch.sqrt(numerator / denominator)

# Relative Squared Error (RSE)
class RRSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(torch.mean( (y_pred - y_true)**2) / torch.sum(y_true**2) )
        return loss
    

# Relative Squared Error (RSE)
class RRSE1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(torch.mean( ( (y_pred - y_true)**2) /(y_true**2) ))
        return loss

class FocalRRSE(FocalLossRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, predictions, targets):
        # Compute the absolute difference between predictions and targets
        errors = torch.sqrt(torch.mean((predictions - targets)**2)/torch.sum(targets**2))
        # Find the bin index for each target value
        bin_indices = torch.bucketize(targets, self.bin_edges[1:], right=True) - 1  # Use `-1` to make index 0-based
        bin_indices[bin_indices<0] = 0
        bin_indices[bin_indices>self.num_bins-1] = self.num_bins-1
        # Assign weights based on bin indices
        target_weights = self.weights[bin_indices]

        # Apply penalty for out-of-bound predictions
        
        out_of_bounds_mask = (predictions < self.min_val) | (predictions > self.max_val)
        target_weights[out_of_bounds_mask] *= self.penalty_factor


        # Calculate the focal loss
        loss = self.alpha * (target_weights * (errors ** self.gamma)).mean()

        return loss


class FocalMAPE(FocalLossRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, predictions, targets):
        # Compute the absolute difference between predictions and targets
        errors = (torch.mean(torch.abs(predictions - targets)/ (torch.abs(targets)+1e-8)))

        # Find the bin index for each target value
        bin_indices = torch.bucketize(targets, self.bin_edges[1:], right=True) - 1  # Use `-1` to make index 0-based
        bin_indices[bin_indices<0] = 0
        bin_indices[bin_indices>self.num_bins-1] = self.num_bins-1
        # Assign weights based on bin indices
        target_weights = self.weights[bin_indices]

        # Apply penalty for out-of-bound predictions
        
        out_of_bounds_mask = (predictions < self.min_val) | (predictions > self.max_val)
        target_weights[out_of_bounds_mask] = self.penalty_factor


        # Calculate the focal loss
        loss = self.alpha * (target_weights * (errors ** self.gamma)).mean()

        return loss

