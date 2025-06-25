from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import numpy as np
import torch

class DatasetNormalizer:
    def __init__(self):
        # Initialize scalers
        self.eigenvalue_scaler = RobustScaler()
        self.displacement_scaler = RobustScaler()
        self.rotation_scaler = StandardScaler()
        self.force_scaler = StandardScaler()
        self.mode_shape_disp_scaler = StandardScaler()
        self.mode_shape_rot_scaler = StandardScaler()
        # Coordinate statistics
        self.coord_min = None
        self.coord_max = None
        
        self.gp_force_scaler = StandardScaler()
        self.gp_stress_scaler = RobustScaler()
        self.axial_stress_mean = None
        self.axial_stress_std = None
        self.axial_stress_max = None
        self.axial_stress_min = None
        self.axial_stress_absmax = None

        # Keep track of data ranges for coordinate scaling
        self.coord_min = None
        self.coord_max = None

        self.eigenvalue_min = None
        self.eigenvalue_max = None
        self.force_min = None
        self.force_max = None
        self.disp_min = None
        self.disp_max = None
        self.ms_min = None
        self.ms_max = None
        self.gps_min = None
        self.gps_max = None
        self.axs_min = None
        self.axs_max = None

    def fit(self, dataset, use_z_coord, use_rotations, use_gp_forces, use_axial_stress, use_mode_shapes_as_features, edge_features, prediction_type):
        eigenvalues = []
        displacements = []
        forces = []
        rotations = []
        mode_shapes_displacements = []
        mode_shapes_rotations = []
        coords = []
        gp_forces = []
        gp_stresses = []
        axial_stresses = []

        # Collect data from dataset (same as before)
        for data in dataset:
            if prediction_type == "buckling":
                if hasattr(data, 'eigenvalue'):
                    eigenvalues.append(data.eigenvalue.item())
                elif len(data.y.shape) == 0 or data.y.shape[-1] == 1:
                    eigenvalues.append(data.y.item())

            node_features = data.x.numpy()
            feature_index = 0

            if use_axial_stress:
                if data.edge_attr.shape[1] == 6:
                    axial_stresses.append(data.edge_attr[:, 4].numpy())
            coord_dim = 3 if use_z_coord else 2
            coords.append(node_features[:, :coord_dim])
            feature_index += coord_dim

            # Skip non-normalized features
            feature_index += 1  # SPC
            # External force features
            force_dim = 3 if use_z_coord else 2
            forces.append(node_features[:, feature_index:feature_index+force_dim])
            feature_index += force_dim
            feature_index += 5  # Shell and stiffener

            if "static" in prediction_type:
                static_data = data.y.numpy()
                if use_z_coord and use_rotations:
                    displacements.append(static_data[:, :3])
                    rotations.append(static_data[:, 3:6])
                elif use_z_coord and not use_rotations:
                    displacements.append(static_data[:, :3])
                elif not use_z_coord and use_rotations:
                    displacements.append(static_data[:, :2])
                    rotations.append(static_data[:, 2:4])
                else:
                    displacements.append(static_data[:, :2])
                
                stress_start = -3
                gp_stresses.append(static_data[:, stress_start:])
            else:
                disp_dim = 3 if use_z_coord else 2
                displacements.append(node_features[:, feature_index:feature_index+disp_dim])
                feature_index += disp_dim

                if use_rotations:
                    rot_dim = 3
                    rotations.append(node_features[:, feature_index:feature_index+rot_dim])
                    feature_index += rot_dim

                gp_stresses.append(node_features[:, feature_index:feature_index+3])
                feature_index += 3

            if use_gp_forces and not "static" in prediction_type:
                gp_forces.append(node_features[:, feature_index:feature_index+8])
                feature_index += 8
            
            if use_mode_shapes_as_features and not prediction_type == "mode_shape":
                mode_shapes_displacements.append(node_features[:, feature_index:feature_index+3])
                feature_index += 3
                if use_rotations:
                    mode_shapes_rotations.append(node_features[:, feature_index:feature_index+3])
                    feature_index += 3
            elif hasattr(data, 'mode_shapes') and data.mode_shapes is not None:
                mode_shapes = data.mode_shapes.numpy()
                mode_shapes_displacements.append(mode_shapes[:, :3])
                if use_rotations:
                    mode_shapes_rotations.append(mode_shapes[:, 3:])

        # Fit scalers
        if prediction_type == "buckling" and eigenvalues:
            eigenvalues = np.array(eigenvalues).reshape(-1, 1)
            self.eigenvalue_scaler.fit(eigenvalues)
            self.eigenvalue_min = np.min(eigenvalues, axis=0)
            self.eigenvalue_max = np.max(eigenvalues, axis=0)
            print(f"Eigenvalue scaling: center={self.eigenvalue_scaler.center_[0]}, scale={self.eigenvalue_scaler.scale_[0]}")
            print(f"Min={self.eigenvalue_min}, Max={self.eigenvalue_max}")

        if displacements:
            displacements = np.concatenate(displacements)
            self.displacement_scaler.fit(displacements)
            self.disp_min = np.min(displacements, axis=0)
            self.disp_max = np.max(displacements, axis=0)
            self.displacement_mean = np.mean(displacements, axis=0)
            self.displacement_std = np.std(displacements, axis=0)
            self.displacement_max = np.amax(displacements, axis=0)
            self.displacement_min = np.amin(displacements, axis=0)
            print(f"Displacement scaling: center={self.displacement_scaler.center_}, scale={self.displacement_scaler.scale_}")
            print(f"Min={self.disp_min}, Max={self.disp_max}")

        if rotations:
            rotations = np.concatenate(rotations)
            self.rotation_scaler.fit(rotations)
            print(f"Rotation scaling: center={self.rotation_scaler.mean_}, scale={self.rotation_scaler.scale_}")

        if forces:
            forces = np.concatenate(forces)
            self.force_scaler.fit(forces)
            self.force_min = np.min(forces, axis=0)
            self.force_max = np.max(forces, axis=0)
            print(f"Force scaling: center={self.force_scaler.mean_}, scale={self.force_scaler.scale_}")
            print(f"Min={self.force_min}, Max={self.force_max}")

        if mode_shapes_displacements:
            mode_shapes_displacements = np.concatenate(mode_shapes_displacements)
            self.mode_shape_disp_scaler.fit(mode_shapes_displacements)
            self.ms_min = np.min(mode_shapes_displacements, axis=0)
            self.ms_max = np.max(mode_shapes_displacements, axis=0)
            print(f"Mode shape displacement scaling: center={self.mode_shape_disp_scaler.mean_}, scale={self.mode_shape_disp_scaler.scale_}")
            print(f"Min={self.ms_min}, Max={self.ms_max}")

        if mode_shapes_rotations:
            mode_shapes_rotations = np.concatenate(mode_shapes_rotations)
            self.mode_shape_rot_scaler.fit(mode_shapes_rotations)
            print(f"Mode shape rotation scaling: center={self.mode_shape_rot_scaler.mean_}, scale={self.mode_shape_rot_scaler.scale_}")

        coords = np.concatenate(coords)
        # self.coord_scaler.fit(coords)
        self.coord_min = np.min(coords, axis=0)
        self.coord_max = np.max(coords, axis=0)
        # print(f"Coordinate scaling: center={self.coord_scaler.mean_}, scale={self.coord_scaler.scale_}")
        print(f"Min={self.coord_min}, Max={self.coord_max}")

        if gp_forces:
            gp_forces = np.concatenate(gp_forces)
            self.gp_force_scaler.fit(gp_forces)
            print(f"GP force scaling: center={self.gp_force_scaler.mean_}, scale={self.gp_force_scaler.scale_}")

        gp_stresses = np.concatenate(gp_stresses)
        self.gps_min = np.min(gp_stresses, axis=0)
        self.gps_max = np.max(gp_stresses, axis=0)
        self.gp_stress_scaler.fit(gp_stresses)
        print(f"GP stress scaling: center={self.gp_stress_scaler.center_}, scale={self.gp_stress_scaler.scale_}")
        print(f"Min={self.gps_min}, Max={self.gps_max}")

        if axial_stresses:
            axial_stresses = np.concatenate(axial_stresses).reshape(-1, 1)
            self.axs_min = np.min(axial_stresses, axis=0)
            self.axs_max = np.max(axial_stresses, axis=0)
            self.axial_stress_mean = np.mean(axial_stresses, axis=0)
            self.axial_stress_std = np.std(axial_stresses, axis=0)
            self.axial_stress_max = np.amax(axial_stresses, axis=0)
            self.axial_stress_min = np.amin(axial_stresses, axis=0)
            self.axial_stress_absmax = np.maximum(np.abs(self.axial_stress_max), np.abs(self.axial_stress_min))
            # self.axial_stress_scaler.fit(axial_stresses)
            # print(f"Axial stress scaling: center={self.axial_stress_scaler.mean_}, scale={self.axial_stress_scaler.scale_}")
            print(f"Min={self.axs_min}, Max={self.axs_max}")

    def normalize_eigenvalue(self, eigenvalue):
        return self.eigenvalue_scaler.transform([[eigenvalue]])[0][0]
    
    def denormalize_eigenvalue(self, eigenvalue):
        # Assuming self.eigenvalue_scaler.scale_ and self.eigenvalue_scaler.mean_ are numpy arrays
        scale = torch.tensor(self.eigenvalue_scaler.scale_, dtype=torch.float32, device=eigenvalue.device)
        center = torch.tensor(self.eigenvalue_scaler.center_, dtype=torch.float32, device=eigenvalue.device)
        
        # Perform the inverse transformation using PyTorch operations
        denormalized = eigenvalue * scale + center
        
        return denormalized
    
    # def normalize_displacement(self, displacement):
    #     if torch.is_tensor(displacement):
    #         device = displacement.device
    #         disp_max = torch.tensor(self.displacement_max, device=device)
    #         disp_min = torch.tensor(self.displacement_min, device=device)
    #         return 2 * displacement / torch.maximum(torch.abs(disp_max), torch.abs(disp_min))
    #     return 2 * displacement / (np.maximum(np.abs(self.displacement_max), np.abs(self.displacement_min)))
    
    # def denormalize_displacement(self, normalized_displacement):
    #     if torch.is_tensor(normalized_displacement):
    #         device = normalized_displacement.device
    #         disp_max = torch.tensor(self.displacement_max, device=device)
    #         disp_min = torch.tensor(self.displacement_min, device=device)
    #         return normalized_displacement * (torch.maximum(torch.abs(disp_max), torch.abs(disp_min))) / 2
    #     return normalized_displacement * (np.maximum(np.abs(self.displacement_max), np.abs(self.displacement_min))) / 2
    
    # def normalize_gp_stresses(self, gp_stresses):
    #     if torch.is_tensor(gp_stresses):
    #         device = gp_stresses.device
    #         gps_max = torch.tensor(self.gps_max, device=device)
    #         gps_min = torch.tensor(self.gps_min, device=device)
    #         return 2 * gp_stresses / torch.maximum(torch.abs(gps_max), torch.abs(gps_min))
    #     return 2 * gp_stresses / (np.maximum(np.abs(self.gps_max), np.abs(self.gps_min)))
    
    # def denormalize_gp_stresses(self, normalized_gp_stresses):
    #     if torch.is_tensor(normalized_gp_stresses):
    #         device = normalized_gp_stresses.device
    #         gps_max = torch.tensor(self.gps_max, device=device)
    #         gps_min = torch.tensor(self.gps_min, device=device)
    #         return normalized_gp_stresses * (torch.maximum(torch.abs(gps_max), torch.abs(gps_min))) / 2
    #     return normalized_gp_stresses * (np.maximum(np.abs(self.gps_max), np.abs(self.gps_min))) / 2
    
     # def normalize_force(self, force):
    #     return self.force_scaler.transform(force.reshape(-1, force.shape[-1]))
    #   
    def normalize_rotation(self, rotation):
        return self.rotation_scaler.transform(rotation.reshape(-1, rotation.shape[-1]))  

    def normalize_mode_shape_disp(self, mode_shapes_displacements):
        return self.mode_shape_disp_scaler.transform(mode_shapes_displacements.reshape(-1, mode_shapes_displacements.shape[-1]))

    def normalize_mode_shape_rot(self, mode_shapes_rotations):
        return self.mode_shape_rot_scaler.transform(mode_shapes_rotations.reshape(-1, mode_shapes_rotations.shape[-1]))

    def normalize_gp_forces(self, gp_forces):
        return self.gp_force_scaler.transform(gp_forces.reshape(-1, gp_forces.shape[-1]))
    ########################################################################################################################

    # STATIC ANALYSIS -- NO NORMALIZATION  
    #   
    # def normalize_coordinates(self, coords):
    #     return coords
    
    # def normalize_force(self, force):
    #     return force
    
    # def normalize_displacement(self, displacement):
    #     return displacement
    
    # def denormalize_displacement(self, normalized_displacement):
    #     return normalized_displacement
    
    # def normalize_gp_stresses(self, gp_stresses):
    #     return gp_stresses
    
    # def denormalize_gp_stresses(self, normalized_gp_stresses):
    #     return normalized_gp_stresses
    
    # STATIC ANALYSIS -- NORMALIZATION

    def normalize_coordinates(self, coords):
        denominator = np.maximum(self.coord_max - self.coord_min, 1e-8)/2
        return (coords) / denominator

    def normalize_force(self, force):
        denominator = np.maximum(self.force_max - self.force_min, 1e-8)/2
        return (force) / denominator
    
    def normalize_displacement(self, displacement):
        return self.displacement_scaler.transform(displacement)
    
    def denormalize_displacement(self, normalized_displacement):
        scale = torch.tensor(self.displacement_scaler.scale_, dtype=torch.float32, device=normalized_displacement.device)
        center = torch.tensor(self.displacement_scaler.center_, dtype=torch.float32, device=normalized_displacement.device)
        
        # Perform the inverse transformation using PyTorch operations
        return normalized_displacement * scale + center
    
    def normalize_gp_stresses(self, gp_stresses):
        return self.gp_stress_scaler.transform(gp_stresses)
    
    def denormalize_gp_stresses(self, normalized_gp_stresses):
        scale = torch.tensor(self.gp_stress_scaler.scale_, dtype=torch.float32, device=normalized_gp_stresses.device)
        mean = torch.tensor(self.gp_stress_scaler.center_, dtype=torch.float32, device=normalized_gp_stresses.device)
        
        return (normalized_gp_stresses) * scale + mean
    
    ########################################################################################################################
    def normalize_axial_stress(self, axial_stress):
        #return (axial_stress - self.axial_stress_mean) / (self.axial_stress_std + 1e-8)
        return (axial_stress / self.axial_stress_absmax)*2
    
    def normalize_edge_features(self, edge_features, use_axial_stress):
        normalized_edge_features = edge_features.copy()
        if use_axial_stress:
            normalized_edge_features[:, 4] = self.normalize_axial_stress(edge_features[:, 4])
        return normalized_edge_features
    
    @staticmethod
    def analyze_gp_stress(gp_stresses):
        all_stresses = np.array(gp_stresses[0])
        for i in range(1,len(gp_stresses)):
            all_stresses = np.append(all_stresses, gp_stresses[i], axis=0)

        import matplotlib.pyplot as plt
        bins=100
        # Set up the plotting area
        plt.figure(figsize=(15, 5))
        x_stress = gp_stresses[:, 0]
        y_stress = gp_stresses[:, 1]
        z_stress = gp_stresses[:, 2]
        # Histogram for x-axis stresses
        plt.subplot(1, 3, 1)
        plt.hist(x_stress, bins=bins, alpha=0.7, color='blue')
        plt.axvline(x_stress.mean(), color='red', linestyle='dashed', linewidth=1, label=f"Mean: {x_stress.mean():.4f}")
        plt.title("X-Axis Stress Histogram")
        plt.xlabel("Stress Value")
        plt.ylabel("Frequency")
        plt.legend()

        # Histogram for y-axis stresses
        plt.subplot(1, 3, 2)
        plt.hist(y_stress, bins=bins, alpha=0.7, color='green')
        plt.axvline(y_stress.mean(), color='red', linestyle='dashed', linewidth=1, label=f"Mean: {y_stress.mean():.4f}")
        plt.title("Y-Axis Stress Histogram")
        plt.xlabel("Stress Value")
        plt.ylabel("Frequency")
        plt.legend()

        # Histogram for z-axis stresses
        plt.subplot(1, 3, 3)
        plt.hist(z_stress, bins=bins, alpha=0.7, color='orange')
        plt.axvline(z_stress.mean(), color='red', linestyle='dashed', linewidth=1, label=f"Mean: {z_stress.mean():.4f}")
        plt.title("Z-Axis Stress Histogram")
        plt.xlabel("Stress Value")
        plt.ylabel("Frequency")
        plt.legend()

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()


        ss = RobustScaler()
        normalized_stresses = ss.fit_transform(all_stresses)
        # Set up the plotting area
        plt.figure(figsize=(15, 5))
        x_stress = normalized_stresses[:, 0]
        y_stress = normalized_stresses[:, 1]
        z_stress = normalized_stresses[:, 2]
        # Histogram for x-axis stresses
        plt.subplot(1, 3, 1)
        plt.hist(x_stress, bins=bins, alpha=0.7, color='blue')
        plt.axvline(x_stress.mean(), color='red', linestyle='dashed', linewidth=1, label=f"Mean: {x_stress.mean():.4f}")
        plt.title("X-Axis Normalized Stress Histogram")
        plt.xlabel("Stress Value")
        plt.ylabel("Frequency")
        # plt.ylim([0,500])
        plt.legend()

        # Histogram for y-axis stresses
        plt.subplot(1, 3, 2)
        plt.hist(y_stress, bins=bins, alpha=0.7, color='green')
        plt.axvline(y_stress.mean(), color='red', linestyle='dashed', linewidth=1, label=f"Mean: {y_stress.mean():.4f}")
        plt.title("Y-Axis Normalized Stress Histogram")
        plt.xlabel("Stress Value")
        plt.ylabel("Frequency")
        # plt.ylim([0,500])
        plt.legend()

        # Histogram for z-axis stresses
        plt.subplot(1, 3, 3)
        plt.hist(z_stress, bins=bins, alpha=0.7, color='orange')
        plt.axvline(z_stress.mean(), color='red', linestyle='dashed', linewidth=1, label=f"Mean: {z_stress.mean():.4f}")
        plt.title("Z-Axis Normalized Stress Histogram")
        plt.xlabel("Stress Value")
        plt.ylabel("Frequency")
        # plt.ylim([0,500])
        plt.legend()

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()


        # Set up the plotting area
        plt.figure(figsize=(15, 5))
        x_stress = normalized_stresses[:, 0]
        y_stress = normalized_stresses[:, 1]
        z_stress = normalized_stresses[:, 2]
        # Histogram for x-axis stresses
        plt.subplot(1, 3, 1)
        plt.hist(x_stress, bins=bins, alpha=0.7, color='blue')
        plt.axvline(x_stress.mean(), color='red', linestyle='dashed', linewidth=1, label=f"Mean: {x_stress.mean():.4f}")
        plt.title("X-Axis Normalized Stress Histogram")
        plt.xlabel("Stress Value")
        plt.ylabel("Frequency")
        plt.ylim([0,500])
        plt.legend()

        # Histogram for y-axis stresses
        plt.subplot(1, 3, 2)
        plt.hist(y_stress, bins=bins, alpha=0.7, color='green')
        plt.axvline(y_stress.mean(), color='red', linestyle='dashed', linewidth=1, label=f"Mean: {y_stress.mean():.4f}")
        plt.title("Y-Axis Normalized Stress Histogram")
        plt.xlabel("Stress Value")
        plt.ylabel("Frequency")
        plt.ylim([0,500])
        plt.legend()

        # Histogram for z-axis stresses
        plt.subplot(1, 3, 3)
        plt.hist(z_stress, bins=bins, alpha=0.7, color='orange')
        plt.axvline(z_stress.mean(), color='red', linestyle='dashed', linewidth=1, label=f"Mean: {z_stress.mean():.4f}")
        plt.title("Z-Axis Normalized Stress Histogram")
        plt.xlabel("Stress Value")
        plt.ylabel("Frequency")
        plt.ylim([0,500])
        plt.legend()

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()


