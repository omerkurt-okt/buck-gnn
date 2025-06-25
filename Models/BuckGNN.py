import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import SAGPooling


class BuckGNN(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels=128, 
                 num_layers=6, pooling_layer='mean', prediction_type="buckling",
                 use_z_coord=False, use_rotations=False, dropout_rate=0.1, model_name = "GraphSAGE_MLP"):
        super(BuckGNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.prediction_type = prediction_type
        self.pooling_layer = pooling_layer
        self.num_layers = num_layers
        self.model_name = model_name
        # Determine output dimension based on prediction type
        if prediction_type == "buckling":
            output_dim = 1
        elif prediction_type == "static_disp":
            # Calculate output dimension for displacements only
            if use_z_coord and use_rotations:
                output_dim = 6  # 3 translations + 3 rotations
            elif use_z_coord and not use_rotations:
                output_dim = 3  # 3 translations
            elif not use_z_coord and use_rotations:
                output_dim = 4  # 2 translations + 2 rotations
            else:
                output_dim = 2  # 2 translations
        elif prediction_type == "static_stress":
            output_dim = 3  # σx, σy, τxy
        elif prediction_type == "mode_shape":
            if use_rotations:
                output_dim = 6  # 3 translations + 3 rotations
            else:
                output_dim = 3  # 3 translations
        
        # Encoder
        if hidden_channels <= 128:
            self.node_encoder = nn.Sequential(
                nn.Linear(num_node_features, 64),
                nn.ReLU(),
                nn.Linear(64, hidden_channels),
            )
            
            self.edge_encoder = nn.Sequential(
                nn.Linear(num_edge_features, 64),
                nn.ReLU(),
                nn.Linear(64, hidden_channels),
            )

            if pooling_layer == 'supernode_with_pooling' and prediction_type == "buckling":
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_channels * 2, 64),  # *2 for concatenated features
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
            else:
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_channels, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
        
        elif hidden_channels >= 256:
            self.node_encoder = nn.Sequential(
                nn.Linear(num_node_features, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, hidden_channels),
            )
            
            self.edge_encoder = nn.Sequential(
                nn.Linear(num_edge_features, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, hidden_channels),
            )

            # Decoder options based on pooling layer
            if pooling_layer == 'supernode_with_pooling' and prediction_type == "buckling":
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_channels * 2, 128),  # *2 for concatenated features
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
            else:
                self.decoder = nn.Sequential(      
                    nn.Linear(hidden_channels, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
            
        # Processor
        if model_name == "EA_GNN_Shared":
            self.shared_gn_block = GraphNetBlock(hidden_channels)
        if model_name == "EA_GNN":
            self.gn_blocks = nn.ModuleList([GraphNetBlock(hidden_channels) for _ in range(num_layers)])
        
        
        # self.shared_GraphSAGE_sum = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels, normalize=True, aggr='sum')
        # Replace GraphNetBlocks with SAGEConv layers


        if model_name == "GraphSage_addAggr_Shared":
            self.shared_graphsage_block = SAGEConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        normalize=True,
                        aggr='add'
                    )
        if model_name == "GraphSage_sumAggr":
            self.sage_blocks_sum = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.sage_mlps = nn.ModuleList()
            for _ in range(num_layers):
                self.sage_blocks_sum.append(
                    SAGEConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        normalize=True,
                        aggr='sum'
                    )
                )
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
                self.sage_mlps.append(nn.Linear(hidden_channels, hidden_channels))
        if model_name == "GraphSage_addAggr":
            self.sage_blocks_add = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.sage_mlps = nn.ModuleList()
            for _ in range(num_layers):
                self.sage_blocks_add.append(
                    SAGEConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        normalize=True,
                        aggr='add'
                    )
                )
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
                self.sage_mlps.append(nn.Linear(hidden_channels, hidden_channels))
        if model_name == "GraphSage_meanAggr":
            self.sage_blocks_mean = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.sage_mlps = nn.ModuleList()
            for _ in range(num_layers):
                self.sage_blocks_mean.append(
                    SAGEConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        normalize=True,
                        aggr='mean'
                    )
                )
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
                self.sage_mlps.append(nn.Linear(hidden_channels, hidden_channels))
        if model_name == "GraphSage_maxAggr":
            self.sage_blocks_max = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.sage_mlps = nn.ModuleList()
            for _ in range(num_layers):
                self.sage_blocks_max.append(
                    SAGEConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        normalize=True,
                        aggr='max'
                    )
                )

                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
                self.sage_mlps.append(nn.Linear(hidden_channels, hidden_channels))
        


        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pooling_mpl = MLPPooling(self.hidden_channels, self.hidden_channels, self.hidden_channels)
        # self.hybrid_pooling = HybridPooling(hidden_channels, hidden_channels)

        if model_name =="GraphSAGE_SAG":
            pool_ratio=0.5
            n_layers_before = num_layers // 2
            n_layers_after = num_layers - n_layers_before
            self.sage_layers_1 = nn.ModuleList()
            self.batch_norms_1 = nn.ModuleList()
            for _ in range(n_layers_before):
                self.sage_layers_1.append(
                    SAGEConv(hidden_channels, hidden_channels, normalize=True, aggr='add')
                )
                self.batch_norms_1.append(nn.BatchNorm1d(hidden_channels))
            
            # SAGPooling layer
            self.pool = SAGPooling(
                hidden_channels,
                ratio=pool_ratio,
                GNN=SAGEConv,  # Use GraphSAGE as scoring network
                aggr='add'
            )
            
            # Second set of GraphSAGE layers
            self.sage_layers_2 = nn.ModuleList()
            self.batch_norms_2 = nn.ModuleList()
            for _ in range(n_layers_after):
                self.sage_layers_2.append(
                    SAGEConv(hidden_channels, hidden_channels, normalize=True, aggr='add')
                )
                self.batch_norms_2.append(nn.BatchNorm1d(hidden_channels))

        if model_name =="EAGNN_SAG":
            pool_ratio=0.5
            n_layers_before = num_layers // 2
            n_layers_after = num_layers - n_layers_before
            self.gnn_layers_1 = nn.ModuleList()
            self.batch_norms_1 = nn.ModuleList()
            for _ in range(n_layers_before):
                self.gnn_layers_1.append(
                    GraphNetBlock(hidden_channels)
                )
            
            # SAGPooling layer
            self.pool = SAGPooling(
                hidden_channels,
                ratio=pool_ratio,
                GNN=SAGEConv,  # Use GraphSAGE as scoring network
                aggr='add'
            )
            
            # Second set of GraphSAGE layers
            self.gnn_layers_2 = nn.ModuleList()
            self.batch_norms_2 = nn.ModuleList()
            for _ in range(n_layers_after):
                self.gnn_layers_2.append(
                    GraphNetBlock(hidden_channels)
                )

    def get_pooling_layer(self, x, edge_index, batch):
        """Apply the selected pooling operation using batch information"""
        if "super" in self.pooling_layer:
            if batch is None:
                # Single graph case
                n_nodes = x.size(0)
                super_idx = n_nodes - 1  # Last node is the super node
                real_nodes = torch.arange(n_nodes - 1, device=x.device)  # All but last node
            else:
                # Multiple graphs case
                super_idx = []
                real_nodes = []
                current_batch = 0
                last_idx = -1
                
                for i, b in enumerate(batch):
                    if b != current_batch:
                        super_idx.append(last_idx)
                        current_batch = b
                    last_idx = i
                super_idx.append(last_idx)  # Add the last batch's super node
                
                # Real nodes are all nodes except super nodes
                real_nodes = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
                real_nodes[super_idx] = False
                real_nodes = torch.where(real_nodes)[0]

        if self.pooling_layer == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling_layer == 'hybrid':
            return self.hybrid_pooling(x, batch)
        elif self.pooling_layer == 'mean_no_super':
            if batch is None:
                return global_mean_pool(x[real_nodes].unsqueeze(0), 
                                     torch.zeros(len(real_nodes), dtype=torch.long, device=x.device))
            else:
                return global_mean_pool(x[real_nodes], batch[real_nodes])
        elif self.pooling_layer == 'supernode_only':
            return x[super_idx]
        elif self.pooling_layer == 'supernode_with_pooling':
            if batch is None:
                pooled = global_mean_pool(x[real_nodes].unsqueeze(0), 
                                        torch.zeros(len(real_nodes), dtype=torch.long, device=x.device))
                super_features = x[super_idx].unsqueeze(0)
            else:
                pooled = global_mean_pool(x[real_nodes], batch[real_nodes])
                super_features = x[super_idx]
            return torch.cat([pooled, super_features], dim=1)
        elif self.pooling_layer == 'mlp':
            return self.pooling_mpl(x, batch)
        elif self.pooling_layer == 'mlp_no_super':
            if batch is None:
                return self.pooling_mpl(
                    x[real_nodes], 
                    torch.zeros(len(real_nodes), dtype=torch.long, device=x.device)
                )
            else:
                return self.pooling_mpl(
                    x[real_nodes], batch[real_nodes]
                )
        else:
            raise ValueError(f"Unknown pooling layer: {self.pooling_layer}")



    def forward(self, x, edge_index, edge_attr, batch=None, mask=None):
        model_name = self.model_name
        # Before encoding, identify real nodes based on original node features
        # The last feature in the original node features indicates if it's a super node
        if "super" in self.pooling_layer:
            is_real_node = x[:, -1] == 0 if x.size(1) > 0 else torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            if batch is not None:
                real_node_batch = batch[is_real_node]
            else:
                real_node_batch = None

        # Encode
        x = self.node_encoder(x)

        # Process using shared GNN block multiple times
        if model_name =="EA_GNN_Shared":
            edge_attr = self.edge_encoder(edge_attr)
            for i in range(self.num_layers):
                x_prev, edge_attr_prev = x, edge_attr
                x, edge_attr = self.shared_gn_block(x, edge_index, edge_attr)
                if i > 0 and i < self.num_layers-1:
                    x = x + x_prev  # Skip connection
                    edge_attr = edge_attr + edge_attr_prev  # Skip connection

                x = self.dropout(x)
                edge_attr = self.dropout(edge_attr)

        if model_name =="GraphSage_addAggr_Shared":
            for i in range(self.num_layers):
                x_prev = x
                # Apply GraphSAGE convolution
                x = self.shared_graphsage_block(x, edge_index)
                # Apply batch normalization
                # x = self.batch_norm (x)
                x = self.relu(x)
                # Apply dropout
                
                # Skip connection
                if i > 0 and i < self.num_layers-1: # Skip the first layer for skip connections
                    x = x + x_prev

                x = self.dropout(x)

        elif model_name == "EAGNN_SAG":
            edge_attr = self.edge_encoder(edge_attr)
            for i, conv in enumerate(self.gnn_layers_1):
                x_prev, edge_attr_prev = x, edge_attr
                x, edge_attr = conv(x, edge_index, edge_attr)
                x = self.dropout(x)
                edge_attr = self.dropout(edge_attr)
                if i > 0:
                    x = x + x_prev  # Skip connection
                    edge_attr = edge_attr + edge_attr_prev  # Skip connection
            x, edge_index, edge_attr, batch, pool_perm, pool_score = self.pool(
                x, edge_index, edge_attr, batch
            )
            for conv in self.gnn_layers_2:
                x_prev, edge_attr_prev = x, edge_attr
                x, edge_attr = conv(x, edge_index, edge_attr)
                x = self.dropout(x)
                edge_attr = self.dropout(edge_attr)
                x = x + x_prev  # Skip connection
                edge_attr = edge_attr + edge_attr_prev  # Skip connection

        elif model_name =="EA_GNN":
            edge_attr = self.edge_encoder(edge_attr)
            # Process
            for i, gn_block in enumerate(self.gn_blocks):
                x_prev, edge_attr_prev = x, edge_attr
                x, edge_attr = gn_block(x, edge_index, edge_attr)

                if i > 0 and i < len(self.gn_blocks)-1:
                    x = x + x_prev  # Skip connection
                    edge_attr = edge_attr + edge_attr_prev  # Skip connection

                x = self.dropout(x)
                edge_attr = self.dropout(edge_attr)
        # Process through SAGEConv layers
        elif model_name =="GraphSage_sumAggr":
            for i, (conv, batch_norm) in enumerate(zip(self.sage_blocks_sum, self.batch_norms)):
                x_prev = x
                # Apply GraphSAGE convolution
                x = conv(x, edge_index)
                # Apply batch normalization
                x = batch_norm(x)
                x = self.relu(x)
                # Apply dropout
                
                # Skip connection
                if i > 0 and i < self.num_layers-1: # Skip the first layer for skip connections
                    x = x + x_prev
                # Process through SAGEConv layers
                x = self.dropout(x)
        elif model_name =="GraphSage_addAggr_woBatchNorm":
            for i, (conv, batch_norm) in enumerate(zip(self.sage_blocks_add, self.batch_norms)):
                x_prev = x
                # Apply GraphSAGE convolution
                x = conv(x, edge_index)
                # Apply batch normalization
                x = self.relu(x)
                # Apply dropout
                # Skip connection
                if i > 0 and i < self.num_layers-1: # Skip the first layer for skip connections
                    x = x + x_prev
                
                x = self.dropout(x)
        elif model_name =="GraphSage_sumAggr_woBatchNorm":
            for i, (conv, batch_norm) in enumerate(zip(self.sage_blocks_sum, self.batch_norms)):
                x_prev = x
                # Apply GraphSAGE convolution
                x = conv(x, edge_index)
                # Apply batch normalization
                x = self.relu(x)
                # Apply dropout
                
                # Skip connection
                if i > 0 and i < self.num_layers-1: # Skip the first layer for skip connections
                    x = x + x_prev
                x = self.dropout(x)
        elif model_name =="GraphSage_addAggr":
            for i, (conv, batch_norm) in enumerate(zip(self.sage_blocks_add, self.batch_norms)):
                x_prev = x
                # Apply GraphSAGE convolution
                x = conv(x, edge_index)
                # Apply batch normalization
                x = batch_norm(x)
                x = self.relu(x)
                # Apply dropout
                
                # Skip connection
                if i > 0 and i < len(self.sage_blocks_add)-1: # Skip the first layer for skip connections
                    x = x + x_prev

                x = self.dropout(x)
        elif model_name =="GraphSage_meanAggr":
            for i, (conv, batch_norm) in enumerate(zip(self.sage_blocks_mean, self.batch_norms)):
                x_prev = x
                # Apply GraphSAGE convolution
                x = conv(x, edge_index)
                # Apply batch normalization
                x = batch_norm(x)
                x = self.relu(x)
                # Apply dropout
                
                # Skip connection
                if i > 0 and i < self.num_layers-1: # Skip the first layer for skip connections
                    x = x + x_prev
                x = self.dropout(x)
        elif model_name =="GraphSage_maxAggr":
            for i, (conv, batch_norm) in enumerate(zip(self.sage_blocks_max, self.batch_norms)):
                x_prev = x
                # Apply GraphSAGE convolution
                x = conv(x, edge_index)
                # Apply batch normalization
                x = batch_norm(x)
                x = self.relu(x)

                # Skip connection
                if i > 0 and i < self.num_layers-1: # Skip the first layer for skip connections
                    x = x + x_prev
                x = self.dropout(x)
        elif model_name =="GraphSage_MLP":
            for i, (conv, batch_norm, mlp) in enumerate(zip(self.sage_blocks_add, self.batch_norms, self.sage_mlps)):
                x_prev = x
                # Apply GraphSAGE convolution
                x = conv(x, edge_index)
                x_sage = x
                # Apply batch normalization
                x = batch_norm(x)
                x = self.relu(x)
                x = mlp(x)
                x = batch_norm(x)
                x = self.relu(x)
                # Apply dropout

                x = x_sage + x
                # Skip connection

                # Skip connection
                if i > 0 and i < self.num_layers-1: # Skip the first layer for skip connections
                    x = x + x_prev
                x = self.dropout(x)
        elif model_name == "GraphSAGE_SAG":
            for i, (conv, batch_norm) in enumerate(zip(self.sage_layers_1, self.batch_norms_1)):
                identity = x
                x = conv(x, edge_index)
                x = batch_norm(x)
                x = self.relu(x)
                x = self.dropout(x)
                if i > 0:
                    x = x + identity  # Skip connection
            x, edge_index, edge_attr, batch, pool_perm, pool_score = self.pool(
                x, edge_index, edge_attr, batch
            )
            for conv, batch_norm in zip(self.sage_layers_2, self.batch_norms_2):
                identity = x
                x = conv(x, edge_index)
                x = batch_norm(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = x + identity  # Skip connection

        if self.prediction_type == "buckling":
            # Use appropriate pooling method
            pooled = self.get_pooling_layer(x, edge_index, batch)
            return self.decoder(pooled).squeeze(), batch
        
        elif "static" in self.prediction_type or "mode_shape" in self.prediction_type:
            if "super" in self.pooling_layer:
                # For static and mode shape predictions, exclude super nodes if they exist
                return self.decoder(x[is_real_node]), real_node_batch
            else:
                # If not using super nodes, predict for all nodes
                return self.decoder(x), batch
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

class GraphNetBlock(nn.Module):
    def __init__(self, hidden_channels):
        super(GraphNetBlock, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.node_mlp_phi = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.node_mlp_gamma = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.node_mlp_beta = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        
        # Edge update
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_attr = self.edge_mlp(edge_features)
        
        # Node update
        messages = self.node_mlp_phi(torch.cat([x[col], edge_attr], dim=1))
        aggregated_messages = scatter_mean(messages, row, dim=0, dim_size=x.size(0))
        node_features = torch.cat([x, aggregated_messages], dim=1)
        x = self.node_mlp_gamma(node_features)
        x = x + self.node_mlp_beta(x)
        
        return x, edge_attr

class MLPPooling(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLPPooling, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            # nn.Linear(hidden_channels, hidden_channels),
            # nn.ReLU()
        )
    def forward(self, x, batch):
        # Global mean pooling
        x = global_mean_pool(x, batch)
        # Apply MLP
        return self.mlp(x)
    
class HybridPooling(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(HybridPooling, self).__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(in_channels * 3, hidden_channels),  # *3 for mean, max, and attention
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )
        
    def forward(self, x, batch):
        # Compute attention weights
        attention_weights = self.attention_mlp(x)
        attention_weights = torch.sigmoid(attention_weights)
        
        # Apply attention-weighted pooling
        weighted_x = x * attention_weights
        attention_pooled = scatter_add(weighted_x, batch, dim=0)
        
        # Mean and max pooling
        mean_pooled = global_mean_pool(x, batch)
        max_pooled = global_max_pool(x, batch)
        
        # Combine all pooled features
        combined = torch.cat([attention_pooled, mean_pooled, max_pooled], dim=1)
        output = self.feature_mlp(combined)
        
        return output