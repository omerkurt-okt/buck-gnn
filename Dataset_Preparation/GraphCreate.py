import os
import numpy as np
import torch
from torch_geometric.data import Data
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import pickle
import matplotlib.pyplot as plt
from Dataset_Preparation.Normalizer import DatasetNormalizer
from Utils.Visualization import visualize_graph
from Dataset_Preparation.Transformation import transform_to_simulation_coordinates, calculate_stiffener_contributions
from Dataset_Preparation.VirtualEdgeCreate import VirtualEdgeCache, add_virtual_edges, create_super_node, create_hybrid_virtual_edges
import sys
import shutil
import json
from datetime import datetime
from Dataset_Preparation.DatasetSplit import detect_buckling_outliers, detect_static_outliers, detect_modeshape_outliers

# Create a global cache instance
# virtual_edge_cache = VirtualEdgeCache(cache_file="virtual_edges_cache.pkl")

def get_unique_shapes(bdf_files):
    """Extract unique shape identifiers from file names, handling both normal and cutout shapes."""
    shape_groups = {}
    
    for bdf_file in bdf_files:
        base_name = os.path.basename(bdf_file)
        parts = base_name.split('_')
        
        # Handle cutout shapes (shape_with_cutout_0000_lc0_pristine.bdf)
        if 'cutout' in base_name:
            shape_id = '_'.join(parts[:4])  # Gets 'shape_with_cutout_0000'
        # Handle normal shapes (shape_0034_lc3_pristine.bdf)
        else:
            shape_id = '_'.join(parts[:2])  # Gets 'shape_0034'
        
        if shape_id not in shape_groups:
            shape_groups[shape_id] = []
        shape_groups[shape_id].append(bdf_file)
    
    # Print summary
    normal_shapes = sum(1 for k in shape_groups if 'cutout' not in k)
    cutout_shapes = sum(1 for k in shape_groups if 'cutout' in k)
    print(f"\nShape summary:")
    print(f"Normal shapes: {normal_shapes}")
    print(f"Cutout shapes: {cutout_shapes}")
    print(f"Total unique shapes: {len(shape_groups)}")
    
    return shape_groups

def parse_nastran_results(op2_file):
    op2 = OP2(debug=False)
    op2.read_op2(op2_file)

    try:
        all_buck_subcases = list(op2.eigenvectors.keys())
        buck_isubcase = all_buck_subcases[0]

        if buck_isubcase not in op2.eigenvectors:
            return None, None, None

        eigenvectors = op2.eigenvectors[buck_isubcase]
        
        if eigenvectors.eigrs is None:
            return None, None, None

        eigenvalue = eigenvectors.eigrs[0]
        mode_shape = op2.eigenvectors[buck_isubcase].data[0] if buck_isubcase in op2.eigenvectors else None

        all_static_subcases = list(op2.displacements.keys())
        static_isubcase = all_static_subcases[0]

        static_displacements = op2.displacements[static_isubcase].data[0] if static_isubcase in op2.displacements else None

        all_gpstress_subcases = list(op2.grid_point_surface_stresses.keys())
        gpstress_isubcase = all_gpstress_subcases[0]

        gp_stresses = op2.grid_point_surface_stresses[gpstress_isubcase].data[0] if gpstress_isubcase in op2.grid_point_surface_stresses else None
        gp_stresses = make_unique_groups(gp_stresses)
        
        all_cbarstress_subcases = list(op2.cbar_stress.keys())
        cbarstress_isubcase = all_cbarstress_subcases[0]
        cbar_stress = op2.cbar_stress[cbarstress_isubcase] if cbarstress_isubcase in op2.cbar_stress else None

        all_gpforce_subcases = list(op2.grid_point_forces.keys())
        gpforce_isubcase = all_gpforce_subcases[0]
        gp_forces = {}

        for i, element_name in enumerate(op2.grid_point_forces[gpforce_isubcase].element_names[0]):
            if element_name.startswith('QUAD4'):
                node_element = op2.grid_point_forces[gpforce_isubcase].node_element[0][i]
                force_data = op2.grid_point_forces[gpforce_isubcase].data[0][i]
                
                node_id = node_element[0]
                element_id = node_element[1]
                
                if node_id not in gp_forces:
                    gp_forces[node_id] = {}
                
                gp_forces[node_id][element_id] = force_data[:3]

        return eigenvalue, static_displacements, mode_shape, gp_stresses, gp_forces, cbar_stress

    except Exception as e:
        logging.exception(f"Error in parse_nastran_results: {str(e)}")
        return None, None, None, None, None, None
    
def find_boundary_nodes(bdf_model):
    """
    Create a set of boundary nodes by analyzing the mesh once.
    """
    edge_counts = {}
    boundary_nodes = set()

    # Create node ID to index mapping
    sorted_node_ids = sorted(bdf_model.nodes.keys())
    node_id_to_index = {nid: i for i, nid in enumerate(sorted_node_ids)}

    # First collect all edges and their counts
    for elem in bdf_model.elements.values():
        if elem.type in ['CQUAD4', 'CTRIA3']:
            nodes = elem.nodes
            # For each element, get its edges
            for i in range(len(nodes)):
                n1, n2 = nodes[i], nodes[(i+1)%len(nodes)]
                # Convert node IDs to indices
                idx1, idx2 = node_id_to_index[n1], node_id_to_index[n2]
                edge = tuple(sorted([idx1, idx2]))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

    # Find boundary nodes (nodes that have at least one edge appearing once)
    for edge, count in edge_counts.items():
        if count == 1:
            boundary_nodes.add(edge[0])
            boundary_nodes.add(edge[1])

    return boundary_nodes

def create_graph_from_bdf(bdf_model, static_displacements, mode_shape, gp_forces, gp_stresses, cbar_stress, 
                         use_z_coord=False, use_rotations=False, use_gp_forces=False, 
                         use_axial_stress=False, use_mode_shapes_as_features=False, 
                         virtual_edges_file=None, use_super_node=False, transform=True,
                         prediction_type="buckling", virtual_edge_cache=None):
    
    # Sort node IDs first for consistent ordering
    sorted_node_ids = sorted(bdf_model.nodes.keys())
    node_id_to_index = {nid: i for i, nid in enumerate(sorted_node_ids)}

    # Create coordinate arrays with consistent ordering
    if use_z_coord:
        node_coords = np.array([bdf_model.nodes[nid].xyz for nid in sorted_node_ids])
    else:
        node_coords = np.array([bdf_model.nodes[nid].xyz[:2] for nid in sorted_node_ids])
    
    # Transform coordinates
    if transform:
        transformed_coords, centroid, transformation_matrix, transform_info= transform_to_simulation_coordinates(node_coords[:, :2])
    else:
        transformed_coords = node_coords[:, :2]
        transformation_matrix = np.eye(2)
    
    # Get boundary nodes using indices
    boundary_nodes = find_boundary_nodes(bdf_model)
    
    node_features = []
    static_targets = []
    
    edges = {}
    nastran_results_mapping = {}

    is_static_analysis = "static" in prediction_type
    is_mode_shape_prediction = prediction_type == "mode_shape"
    
    for i, node_id in enumerate(sorted_node_ids):
        node = bdf_model.nodes[node_id]
        nastran_results_mapping[node_id] = i
        # Start with transformed coordinates
        if use_z_coord:
            node_feature = list(transformed_coords[i]) + [node_coords[i, 2]]  # Keep Z-coordinate unchanged
        else:
            node_feature = list(transformed_coords[i])


        # SPC features (no transformation needed - it is binary)
        spc_feature = [0]
        for spc in bdf_model.spcs[1]:
            if node_id in spc.node_ids:
                if spc.components == '123456':
                    spc_feature = [1]
                else:
                    spc_feature = [0.25]
        node_feature.extend(spc_feature)

        # Transform force features
        force_feature = [0, 0, 0] if use_z_coord else [0, 0]
        for force in bdf_model.loads[2]:
            if node_id == force.node_id:
                if use_z_coord:
                    force_vec = np.array(force.scaled_vector)
                    force_vec[:2] = force_vec[:2] @ transformation_matrix
                    force_vec = force_vec  # Normalize by max force
                    force_feature = list(force_vec)
                else:
                    force_vec = np.array(force.scaled_vector[:2])
                    transformed_force = force_vec @ transformation_matrix
                    transformed_force = transformed_force  # Normalize by max force
                    force_feature = list(transformed_force)
        node_feature.extend(force_feature)

        # Calculate transformed stiffness contributions for CBAR elements
        connected_cbars = [elem for elem in bdf_model.elements.values() 
                         if elem.type == 'CBAR' and node_id in elem.nodes]
        
        stiffener_bins = calculate_stiffener_contributions(
            node_id,
            connected_cbars, 
            np.array(node.xyz[:2]), 
            bdf_model, 
            transformation_matrix
        )

        # Use index-based boundary node check
        is_boundary = float(i in boundary_nodes)
        stiffener_bins = [bin_value/3.0 for bin_value in stiffener_bins]

        # Add boundary indicator and scaled stiffener bins to node features
        node_feature.extend([is_boundary] + stiffener_bins)

        
        static_target = list()
        # Transform displacement features
        if static_displacements is not None:
            
            if use_z_coord:
                disp = np.array(static_displacements[i])  # Keep original indexing for nastran results
                disp[:2] = disp[:2] @ transformation_matrix
                node_feature.extend(disp[:3])
                if use_rotations:
                    rot = disp[3:]
                    rot[:2] = rot[:2] @ transformation_matrix
                    node_feature.extend(rot)
            else:
                disp = np.array(static_displacements[i][:2])
                transformed_disp = disp @ transformation_matrix
                if not is_static_analysis:
                    node_feature.extend(transformed_disp)
                else:
                    static_target.extend(transformed_disp)

        # Transform GP stresses using Mohr's circle
        if gp_stresses is not None:
            σx = gp_stresses[i][0]
            σy = gp_stresses[i][1]
            τxy = gp_stresses[i][2]
            if transform:
                if transform_info is None:
                    θ = -np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
                else:
                    θ = -transform_info['rotation_angle']
            else:
                θ = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])

            # θ_deg = np.degrees(θ)

            # Transform stresses using Mohr's circle equations
            σx_new = (σx + σy)/2 + (σx - σy)/2 * np.cos(2*θ) + τxy * np.sin(2*θ)
            σy_new = (σx + σy)/2 - (σx - σy)/2 * np.cos(2*θ) - τxy * np.sin(2*θ)
            τxy_new = -(σx - σy)/2 * np.sin(2*θ) + τxy * np.cos(2*θ)
            
            # Apply flipping transformations
            if transform:
                if transform_info is not None:
                    flip_x = transform_info['flip_x']
                    flip_y = transform_info['flip_y']
                    # Under axis flipping:
                    # - If x-axis is flipped: σx stays same, τxy changes sign
                    # - If y-axis is flipped: σy stays same, τxy changes sign
                    # - If both axes are flipped: both σx and σy stay same, τxy stays same
                    
                    if flip_x != flip_y:  # Only one axis is flipped
                        τxy_new = -τxy_new
            if not is_static_analysis:
                node_feature.extend([σx_new, σy_new, τxy_new])
            else:
                static_target.extend([σx_new, σy_new, τxy_new])
                
        if use_gp_forces and not is_static_analysis and node_id in gp_forces:
            force_sums = np.zeros((4, 2))
            quadrant_counts = np.zeros(4)
            node_coords2 = transformed_coords[i]
            
            for element_id, force in gp_forces[node_id].items():
                element = bdf_model.elements[element_id]
                element_nodes_indices = [node_id_to_index[n] for n in element.nodes]
                element_center = np.mean([transformed_coords[idx] for idx in element_nodes_indices], axis=0)
                
                relative_pos = element_center - node_coords2
                quadrant = (int(relative_pos[0] < 0) * 2) + int(relative_pos[1] < 0)
                
                force_vec = np.array(force[:2])
                transformed_force = force_vec @ transformation_matrix
                
                force_sums[quadrant] += transformed_force
                quadrant_counts[quadrant] += 1
            
            force_features = []
            for quadrant in range(4):
                if quadrant_counts[quadrant] > 0:
                    avg_force = force_sums[quadrant] / quadrant_counts[quadrant]
                else:
                    avg_force = np.zeros(2)
                force_features.extend(avg_force)
            
            node_feature.extend(force_features)

        # Transform mode shape features
        if mode_shape is not None and use_mode_shapes_as_features and not is_static_analysis:
            mode = np.array(mode_shape[i])
            mode[:2] = mode[:2] @ transformation_matrix
            node_feature.extend(mode[:3])
            if use_rotations:
                mode_rot = mode[3:]
                mode_rot[:2] = mode_rot[:2] @ transformation_matrix
                node_feature.extend(mode_rot)
        
        node_features.append(node_feature)
        if "static" in prediction_type:
            static_targets.append(static_target)
    # Create edges (using transformed coordinates)
    for elem_id, elem in bdf_model.elements.items():
        if elem.type == 'CQUAD4':
            for i in range(4):
                n1, n2 = elem.nodes[i], elem.nodes[(i+1)%4]
                idx1, idx2 = node_id_to_index[n1], node_id_to_index[n2]
                
                pos1 = transformed_coords[idx1]
                pos2 = transformed_coords[idx2]
                
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                distance = np.sqrt(dx**2 + dy**2)
                direction = np.array([dx, dy]) / distance
                
                edge = tuple(sorted([idx1, idx2]))
                if edge not in edges:
                    edges[edge] = [0.01, distance/1000, direction[0], direction[1]]

        elif elem.type == 'CBAR':
            n1, n2 = elem.nodes
            idx1, idx2 = node_id_to_index[n1], node_id_to_index[n2]
            
            pos1 = transformed_coords[idx1]
            pos2 = transformed_coords[idx2]
            
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            distance = np.sqrt(dx**2 + dy**2)
            direction = np.array([dx, dy]) / distance
            
            edge = tuple(sorted([idx1, idx2]))
            
            if elem.pid == 900:
                edges[edge] = [1, distance/1000, direction[0], direction[1]]
            else:
                edges[edge] = [0.01, distance/1000, direction[0], direction[1]]

            if use_axial_stress and cbar_stress is not None and not is_static_analysis:
                if elem_id in cbar_stress.element:
                    stress_idx = np.where(cbar_stress.element == elem_id)[0][0]
                    axial_stress = cbar_stress.data[0, stress_idx, 4]
                    edges[edge].extend([axial_stress])
                else:
                    edges[edge].extend([0])

    if virtual_edges_file is not None and not use_super_node:
        if os.path.exists(virtual_edges_file):
            print(f"Loading virtual edges from file: {virtual_edges_file}")
            with open(virtual_edges_file, 'r') as f:
                virtual_edges = []
                for line in f:
                    node1, node2 = map(int, line.strip().split(','))
                    if node1 in node_id_to_index and node2 in node_id_to_index:
                        edge = tuple(sorted([node_id_to_index[node1], node_id_to_index[node2]]))
                        virtual_edges.append(edge)
        else:
            print("Creating hybrid virtual edges...")
            if virtual_edge_cache is None:
                virtual_edge_cache = VirtualEdgeCache()
            virtual_edges = virtual_edge_cache.get_virtual_edges(transformed_coords, edges)
        edges = add_virtual_edges(edges, virtual_edges, transformed_coords, use_axial_stress)
    
    elif virtual_edges_file is None and use_super_node is False:
        print("Creating hybrid virtual edges...")
        if virtual_edge_cache is None:
            virtual_edge_cache = VirtualEdgeCache()
        virtual_edges = virtual_edge_cache.get_virtual_edges(transformed_coords, edges)
        edges = add_virtual_edges(edges, virtual_edges, transformed_coords, use_axial_stress)
    
    elif use_super_node:
        print("Adding super node...")
        super_node_pos = np.array([0.0, 0.0])
        if use_z_coord:
            super_node_pos = np.append(super_node_pos, 0.0)
        transformed_coords = np.vstack([transformed_coords, super_node_pos])
        node_features, super_node_features, super_node_edges = create_super_node(
            node_features, 
            edges, 
            len(node_features[0])
        )
        node_features.append(super_node_features)
        edges = add_virtual_edges(edges, super_node_edges, transformed_coords, use_axial_stress)

    # Create edge tensors
    edge_index = []
    edge_features = []
    for (n1, n2), feature in edges.items():
        edge_index.extend([[n1, n2], [n2, n1]])
        edge_features.extend([feature, feature])
    if transform is False:
        transform_info=None
    return (torch.tensor(node_features, dtype=torch.float), 
            torch.tensor(edge_index, dtype=torch.long).t().contiguous(), 
            torch.tensor(edge_features, dtype=torch.float),
            transformation_matrix,
            transform_info,
            transform,
            torch.tensor(static_targets, dtype=torch.float) if "static" in prediction_type else None
            )

def log_problem(problems, log_file, file_path, error_msg):
    """Log problematic file with timestamp and error message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_name = os.path.basename(file_path)
    
    problems[base_name] = {
        "timestamp": timestamp,
        "error": error_msg,
        "full_path": file_path
    }
    
    with open(log_file, 'w') as f:
        json.dump(problems, f, indent=4)
    
    print(f"\nProblem detected in {base_name}:")
    print(f"Error: {error_msg}")

def move_problematic_files(bdf_path, op2_path, problem_dir):
    """Move problematic BDF and OP2 files to separate directory."""
    try:
        if os.path.exists(bdf_path):
            shutil.move(bdf_path, os.path.join(problem_dir, os.path.basename(bdf_path)))
        if os.path.exists(op2_path):
            shutil.move(op2_path, os.path.join(problem_dir, os.path.basename(op2_path)))
    except Exception as e:
        print(f"Error moving problematic files: {str(e)}")

def load_single_data(args):
    if isinstance(args, tuple) and len(args) == 3:
        file_path, prediction_type, kwargs = args
    else:
        print(f"Reading args")
        sys.exit(0)
        return None

    bdf_path = file_path
    op2_path = file_path.replace(".bdf", ".op2")
    
    # Create problematic files directory if it doesn't exist
    problem_dir = os.path.join(os.path.dirname(bdf_path), "problematic_files")
    os.makedirs(problem_dir, exist_ok=True)
    problem_log = os.path.join(problem_dir, "problem_log.json")
    
    # # Load existing problem log if it exists
    # if os.path.exists(problem_log):
    #     with open(problem_log, 'r') as f:
    #         problems = json.load(f)
    # else:
    #     problems = {}
    problems = {}

    if not os.path.exists(op2_path):
        print(f"Warning: OP2 file not found for {bdf_path}")
        return None
    
    bdf_model = BDF(debug=False)
    bdf_model.read_bdf(bdf_path)
    
    try:
        eigenvalue, static_displacements, mode_shape, gp_stresses, gp_forces, cbar_stress = parse_nastran_results(op2_path)
    except Exception as e:
        print(f"Error parsing OP2 file {op2_path}: {str(e)}")
        return None
    
    if eigenvalue is None or static_displacements is None or gp_stresses is None:
        return None

    # Get node count before graph creation
    num_nodes = len(bdf_model.nodes)
    disp_nodes = len(static_displacements)
    stress_nodes = len(gp_stresses)
    if disp_nodes != num_nodes or stress_nodes != num_nodes:
        error_msg = (f"Node count mismatch - Model: {num_nodes}, "
                    f"Displacements: {disp_nodes}, Stresses: {stress_nodes}")
        log_problem(problems, problem_log, bdf_path, error_msg)
        
        # Move problematic files to separate directory
        move_problematic_files(bdf_path, op2_path, problem_dir)
        return None
    
    
    virtual_edge_cache = VirtualEdgeCache(cache_file=kwargs.get('virtual_edge_cache'))
    cache_file = kwargs.pop('virtual_edge_cache', None)  # Remove from kwargs
    node_features, edge_index, edge_features, transformation_matrix, transform_info, transform, static_targets = create_graph_from_bdf(
        bdf_model, static_displacements, mode_shape, gp_forces, gp_stresses, cbar_stress,
        prediction_type=prediction_type, virtual_edge_cache=virtual_edge_cache,
        **kwargs
    )
    if not transform:
        transformation_matrix = np.eye(2)
    # Create appropriate target based on prediction type
    if prediction_type == "buckling":
        target = torch.tensor([eigenvalue])
    elif "static" in prediction_type:
        target = static_targets
    elif prediction_type == "mode_shape":
        use_z_coord = kwargs.get('use_z_coord', False)
        use_rotations = kwargs.get('use_rotations', False)
            # Transform mode shapes
        mode_shape_data = np.array(mode_shape)
        if use_rotations:
            # Transform translations and rotations
            mode_shape_data[:, :2] = mode_shape_data[:, :2] @ transformation_matrix
            mode_shape_data[:, 3:5] = mode_shape_data[:, 3:5] @ transformation_matrix
            target = torch.tensor(mode_shape_data, dtype=torch.float)  # Full 6 DOF
        else:
            # Transform translations only
            mode_shape_data[:, :2] = mode_shape_data[:, :2] @ transformation_matrix
            target = torch.tensor(mode_shape_data[:, :3], dtype=torch.float)  # 3 translations

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=target)
    data.file_path = file_path
    
    # Store eigenvalue as additional attribute for mode shape prediction
    if prediction_type == "mode_shape":
        data.eigenvalue = torch.tensor([eigenvalue])

    if prediction_type == "buckling":
        data.mode_shapes = torch.tensor(mode_shape)

    return data

def load_dataset_parallel(data_dir, use_z_coord, use_rotations, use_gp_forces, 
                         use_axial_stress, use_mode_shapes_as_features, 
                         virtual_edges_file, use_super_node,
                         prediction_type="buckling",
                         num_processes=mp.cpu_count()-2):
    
    dataset_cache_file = os.path.join(data_dir, f'dataset_cache_{prediction_type if not "static" in prediction_type else "static"}.pkl')
    virtual_edge_cache_file = os.path.join(data_dir, "virtual_edges_cache.pkl")
    
    if os.path.exists(dataset_cache_file):
        print("Loading cached dataset...")
        with open(dataset_cache_file, "rb") as f:
            return pickle.load(f)

    print("Loading dataset...")
    bdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".bdf")]
    print(f"Found {len(bdf_files)} total BDF files")

    # Only process unique shapes if we're using automatic virtual edges
    # if not use_super_node and virtual_edges_file is None:
    #     # Group files by unique shapes
    #     shape_groups = get_unique_shapes(bdf_files)
    #     print("\nShape summary:")
    #     print(f"Normal shapes: {sum(1 for k in shape_groups if 'cutout' not in k)}")
    #     print(f"Cutout shapes: {sum(1 for k in shape_groups if 'cutout' in k)}")
    #     print(f"Total unique shapes: {len(shape_groups)}")
        
    #     # Process unique shapes first using all threads
    #     print("\nProcessing unique shapes...")
    #     unique_shape_files = [group[0] for group in shape_groups.values()]
        
    #     # Create arguments for each unique shape
    #     process_args = [(f, virtual_edge_cache_file, {'use_z_coord': use_z_coord}) 
    #                    for f in unique_shape_files]
        
    #     with mp.Pool(processes=num_processes) as pool:
    #         unique_results = list(tqdm(
    #             pool.imap(process_unique_shape, process_args),
    #             total=len(unique_shape_files),
    #             desc="Processing unique shapes"
    #         ))
            
    #     # Merge caches after processing unique shapes
    #     VirtualEdgeCache.merge_caches(virtual_edge_cache_file)
        
    #     print("\nUnique shapes processed, now processing all files...")
    # else:
    #     print("\nSkipping unique shape processing:")
    #     if use_super_node:
    #         print("- Using super node configuration")
    #     if virtual_edges_file:
    #         print(f"- Using virtual edges from file: {virtual_edges_file}")
    # sys.exit(0)
    # Process all files
    process_args = [(
        bdf_file, 
        prediction_type, 
        dict(
            use_z_coord=use_z_coord,
            use_rotations=use_rotations,
            use_gp_forces=use_gp_forces,
            use_axial_stress=use_axial_stress,
            use_mode_shapes_as_features=use_mode_shapes_as_features,
            virtual_edges_file=virtual_edges_file,
            use_super_node=use_super_node,
            virtual_edge_cache=virtual_edge_cache_file
        )
    ) for bdf_file in bdf_files]
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(load_single_data, process_args),  # Changed from starmap to imap
            total=len(bdf_files),
            desc="Processing all files"
        ))
    dataset = [r for r in results if r is not None]
    
    print("\nProcessing complete:")
    print(f"Successfully loaded {len(dataset)} out of {len(bdf_files)} files")
    
    print("\nCaching dataset...")
    with open(dataset_cache_file, "wb") as f:
        pickle.dump(dataset, f)
    
    return dataset

def print_displacement_check(dataset, prediction_type="static"):
    """Print information about graphs with extreme displacements."""
    print("\nChecking extreme displacements...")
    
    extreme_cases = []
    for i, data in enumerate(dataset):
        # Check if y is 1D or 2D
        if len(data.y.shape) == 1:
            # For 1D data (like eigenvalues), skip
            continue
            
        # For 2D data (displacements and stresses)
        disps = data.y.numpy()[:, :2]  # Get 2D displacements
        max_disp = np.max(np.abs(disps))
        
        if max_disp > 100:  # threshold for extreme displacement
            extreme_cases.append({
                'index': i,
                'max_abs': max_disp,
                'x_range': [np.min(disps[:, 0]), np.max(disps[:, 0])],
                'y_range': [np.min(disps[:, 1]), np.max(disps[:, 1])]
            })
    
    # Print extreme cases
    for case in extreme_cases:
        print(f"\nGraph {case['index']}:")
        print(f"  Max absolute displacement: {case['max_abs']:.2f}")
        print(f"  Displacement ranges:")
        print(f"    X: [{case['x_range'][0]:.2f}, {case['x_range'][1]:.2f}]")
        print(f"    Y: [{case['y_range'][0]:.2f}, {case['y_range'][1]:.2f}]")



def dataset_normalizer(dataset, use_z_coord=False, use_rotations=False, use_gp_forces=False, 
                use_axial_stress=False, use_mode_shapes_as_features=False,
                data_edge_attr=None, prediction_type="buckling",normalizer=None):
    if normalizer==None:
        normalizer = DatasetNormalizer()
        normalizer.fit(dataset, use_z_coord, use_rotations, use_gp_forces, use_axial_stress, use_mode_shapes_as_features, data_edge_attr, prediction_type)
    else:
        normalizer=normalizer

    normalized_dataset = []
    for data in dataset:
        normalized_data = data.clone()
        
        # Normalize node features
        node_features = data.x.numpy()
        normalized_node_features = np.zeros_like(node_features)
        
        # Check which nodes are super nodes (last feature is 1 for super nodes)
        is_super_node = node_features[:, -1] == 1
        
        feature_index = 0
        
        # Coordinates
        coord_dim = 3 if use_z_coord else 2
        normalized_node_features[:, :coord_dim] = normalizer.normalize_coordinates(node_features[:, :coord_dim])
        feature_index += coord_dim

        # Skip normalization for SPC features (1 features)
        normalized_node_features[:, feature_index:feature_index+1] = node_features[:, feature_index:feature_index+1]
        feature_index += 1

        # External forces
        ex_force_dim = 3 if use_z_coord else 2
        normalized_node_features[:, feature_index:feature_index+ex_force_dim] = normalizer.normalize_force(node_features[:, feature_index:feature_index+ex_force_dim])
        feature_index += ex_force_dim

        # Skip normalization for shell and stiffener features (5 features)
        normalized_node_features[:, feature_index:feature_index+5] = node_features[:, feature_index:feature_index+5]
        feature_index += 5

        if not "static" in prediction_type:
            # Only normalize these features for buckling analysis
            disp_dim = 3 if use_z_coord else 2
            normalized_node_features[:, feature_index:feature_index+disp_dim] = normalizer.normalize_displacement(node_features[:, feature_index:feature_index+disp_dim])
            feature_index += disp_dim

            if use_rotations:
                rot_dim = 3
                normalized_node_features[:, feature_index:feature_index+rot_dim] = normalizer.normalize_rotation(node_features[:, feature_index:feature_index+rot_dim])
                feature_index += rot_dim

            # GP stress features
            normalized_node_features[:, feature_index:feature_index+3] = normalizer.normalize_gp_stresses(node_features[:, feature_index:feature_index+3])
            feature_index += 3

        if use_gp_forces and not "static" in prediction_type:
            normalized_node_features[:, feature_index:feature_index+8] = normalizer.normalize_gp_forces(node_features[:, feature_index:feature_index+8])
            feature_index += 8
        
        if use_mode_shapes_as_features and not "static" in prediction_type:
            mode_dim = 3
            normalized_node_features[:, feature_index:feature_index+mode_dim] = normalizer.normalize_mode_shape_disp(node_features[:, feature_index:feature_index+mode_dim])
            feature_index += mode_dim
            if use_rotations:
                normalized_node_features[:, feature_index:feature_index+mode_dim] = normalizer.normalize_mode_shape_rot(node_features[:, feature_index:feature_index+mode_dim])
                feature_index += mode_dim
        
        # Ensure super nodes remain zero for all features except the indicator
        normalized_node_features[is_super_node] = 0
        normalized_node_features[is_super_node, -1] = 1  # Keep super node indicator as 1
        normalized_data.x = torch.FloatTensor(normalized_node_features)
        
        # Normalize targets based on prediction type
        if prediction_type == "buckling":
            normalized_data.y = torch.FloatTensor([normalizer.normalize_eigenvalue(data.y.item())])
        elif "static" in prediction_type:
            # Split target into displacement and stress components
            disp_dim = data.y.shape[1] - 3  # Last 3 components are stresses
            normalized_disp = normalizer.normalize_displacement(data.y[:, :disp_dim].numpy())
            normalized_stress = normalizer.normalize_gp_stresses(data.y[:, disp_dim:].numpy())
            normalized_data.y = torch.FloatTensor(np.concatenate([normalized_disp, normalized_stress], axis=1))
        elif prediction_type == "mode_shape":
            if use_rotations:
                # Normalize full 6 DOF mode shapes
                mode_shapes = data.y.numpy()
                normalized_mode_shapes = np.zeros_like(mode_shapes)
                normalized_mode_shapes[:, :3] = normalizer.normalize_mode_shape_disp(mode_shapes[:, :3])
                normalized_mode_shapes[:, 3:] = normalizer.normalize_mode_shape_rot(mode_shapes[:, 3:])
            else:
                # Normalize 2D translations
                normalized_mode_shapes = normalizer.normalize_mode_shape_disp(data.y.numpy())
            
            normalized_data.y = torch.FloatTensor(normalized_mode_shapes)
            if hasattr(data, 'eigenvalue'):
                normalized_data.eigenvalue = torch.FloatTensor([normalizer.normalize_eigenvalue(data.eigenvalue.item())])
        
        # Normalize edge features
        if use_axial_stress and not "static" in prediction_type:
            normalized_data.edge_attr = torch.FloatTensor(
                normalizer.normalize_edge_features(data.edge_attr.numpy(), use_axial_stress)
            )
        else:
            normalized_data.edge_attr = data.edge_attr

        if prediction_type == "buckling" and hasattr(data, 'mode_shapes'):
            mode_shapes = data.mode_shapes.numpy()
            normalized_mode_shapes = np.zeros_like(mode_shapes)
            normalized_mode_shapes[:, :3] = normalizer.normalize_mode_shape_disp(mode_shapes[:, :3])
            if use_rotations:
                normalized_mode_shapes[:, 3:] = normalizer.normalize_mode_shape_rot(mode_shapes[:, 3:])
            normalized_data.mode_shapes = torch.FloatTensor(normalized_mode_shapes)

        # visualize_graph(normalized_data.x.numpy(), normalized_data.edge_index.numpy(), normalized_data.edge_attr.numpy())
        normalized_dataset.append(normalized_data)
    return normalized_dataset, normalizer


def load_folder_dataset(data_dir, normalizer, use_z_coord=False, use_rotations=False, use_gp_forces=False, 
                use_axial_stress=False, use_mode_shapes_as_features=False,
                use_super_node=False, virtual_edges_file=None, prediction_type="buckling"):
    
    dataset = load_dataset_parallel(data_dir, use_z_coord, use_rotations, use_gp_forces, 
                                  use_axial_stress, use_mode_shapes_as_features, 
                                  virtual_edges_file, use_super_node, prediction_type)

    print(f"Dataset loaded. Number of graphs: {len(dataset)}")

    # Only check displacements for static analysis
    if "static" in prediction_type:
        print_displacement_check(dataset, prediction_type)

    for i, data in enumerate(dataset[:5]):  # Print info for the first 5 graphs
        # visualize_graph(data.x.numpy(), data.edge_index.numpy(), data.edge_attr.numpy())
        print(f"Graph {i}:")
        print(f"  Node features: {data.x.shape}")
        print(f"  Edge index: {data.edge_index.shape}")
        print(f"  Edge attributes: {data.edge_attr.shape}")
        print(f"  Target shape: {data.y.shape}")
        if "static" in prediction_type:
            print(f"  Target structure:")
            if use_z_coord and use_rotations:
                print(f"    Displacements (translations + rotations): {data.y.shape[1]-3} components")
            elif use_z_coord and not use_rotations:
                print(f"    Displacements (translations only): {data.y.shape[1]-3} components")
            elif not use_z_coord and use_rotations:
                print(f"    Displacements (2D translations + rotations): {data.y.shape[1]-3} components")
            else:
                print(f"    Displacements (2D translations only): {data.y.shape[1]-3} components")
            print(f"    Stresses: 3 components (σx, σy, τxy)")
        if hasattr(data, 'mode_shapes'):
            print(f"  Mode shapes: {data.mode_shapes.shape}")

    if(normalizer==None):
        normalized_dataset, normalizer = dataset_normalizer(dataset, use_z_coord, use_rotations, use_gp_forces, use_axial_stress, use_mode_shapes_as_features, data.edge_attr.numpy(), prediction_type, None)    
        return normalized_dataset, normalizer
    # elif(normalizer == 'cache'):
    #     return dataset
    elif(normalizer):
        normalized_dataset, _ = dataset_normalizer(dataset, use_z_coord, use_rotations, use_gp_forces, use_axial_stress, use_mode_shapes_as_features, data.edge_attr.numpy(), prediction_type, normalizer)    
        return normalized_dataset 
    else:
        raise RuntimeError("Normalizer type not true!")


def load_dataset(data_dir, use_z_coord=False, use_rotations=False, use_gp_forces=False, 
                use_axial_stress=False, use_mode_shapes_as_features=False,
                use_super_node=False, virtual_edges_file=None, prediction_type="buckling"):
    
    dataset = load_dataset_parallel(data_dir, use_z_coord, use_rotations, use_gp_forces, 
                                  use_axial_stress, use_mode_shapes_as_features, 
                                  virtual_edges_file, use_super_node, prediction_type)

    print(f"Dataset loaded. Number of graphs: {len(dataset)}")
    
    # Remove outliers based on prediction type
    # if prediction_type == "buckling":
    #     mask = detect_buckling_outliers(dataset,log_dir="logs/dataset_splits")
    #     dataset = [data for i, data in enumerate(dataset) if mask[i]]
    # elif "static" in prediction_type:
    #     mask = detect_static_outliers(dataset,log_dir="logs/dataset_splits")
    #     dataset = [data for i, data in enumerate(dataset) if mask[i]]
    # elif prediction_type == "mode_shape":
    #     mask = detect_modeshape_outliers(dataset,log_dir="logs/dataset_splits")
    #     dataset = [data for i, data in enumerate(dataset) if mask[i]]
    
    print(f"Dataset after outlier removal. Number of graphs: {len(dataset)}")

    # Only check displacements for static analysis
    if "static" in prediction_type:
        print_displacement_check(dataset, prediction_type)

    for i, data in enumerate(dataset[:5]):  # Print info for the first 5 graphs
        # visualize_graph(data.x.numpy(), data.edge_index.numpy(), data.edge_attr.numpy())
        print(f"Graph {i}:")
        print(f"  Node features: {data.x.shape}")
        print(f"  Edge index: {data.edge_index.shape}")
        print(f"  Edge attributes: {data.edge_attr.shape}")
        print(f"  Target shape: {data.y.shape}")
        if "static" in prediction_type:
            print(f"  Target structure:")
            if use_z_coord and use_rotations:
                print(f"    Displacements (translations + rotations): {data.y.shape[1]-3} components")
            elif use_z_coord and not use_rotations:
                print(f"    Displacements (translations only): {data.y.shape[1]-3} components")
            elif not use_z_coord and use_rotations:
                print(f"    Displacements (2D translations + rotations): {data.y.shape[1]-3} components")
            else:
                print(f"    Displacements (2D translations only): {data.y.shape[1]-3} components")
            print(f"    Stresses: 3 components (σx, σy, τxy)")
        if hasattr(data, 'mode_shapes'):
            print(f"  Mode shapes: {data.mode_shapes.shape}")

    normalized_dataset, normalizer = dataset_normalizer(dataset, use_z_coord, use_rotations, use_gp_forces, use_axial_stress, use_mode_shapes_as_features, data.edge_attr.numpy(), prediction_type)    
        
    return normalized_dataset, normalizer

def make_unique_groups(input_array):
    if input_array.shape[0] % 3 != 0:
        raise ValueError("Number of rows must be a multiple of 3")

    grouped = input_array.reshape(-1, 3, input_array.shape[1])
    dtype = [('row1', grouped.dtype, grouped.shape[2]),
             ('row2', grouped.dtype, grouped.shape[2]),
             ('row3', grouped.dtype, grouped.shape[2])]
    structured = np.array([tuple(group) for group in grouped], dtype=dtype)

    unique_groups, indices = np.unique(structured, return_index=True)
    indices.sort()
    first_elements = [grouped[indice][0] for indice in indices]
    first_elements_array = np.array(first_elements)

    return first_elements_array

def check_graph_transformation(data_dir, num_samples=5, use_z_coord=False, use_rotations=False, 
                             use_gp_forces=False, use_axial_stress=False, 
                             use_mode_shapes_as_features=False, prediction_type="buckling"):
    bdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".bdf")]
    
    for file_idx, bdf_file in enumerate(bdf_files[:num_samples]):
        print(f"\n{'='*80}")
        print(f"Processing file: {bdf_file}")
        print(f"{'='*80}")
        
        # Load models
        bdf_model = BDF(debug=False)
        bdf_model.read_bdf(bdf_file)
        
        op2_file = bdf_file.replace(".bdf", ".op2")
        eigenvalue, static_displacements, mode_shape, gp_stresses, gp_forces, cbar_stress = parse_nastran_results(op2_file)
        
        # Create graphs with and without transformation
        node_features_orig, edge_index_orig, edge_features_orig, transformation_matrix, transform_info, transform, static_targets = create_graph_from_bdf(
            bdf_model, static_displacements, mode_shape, gp_forces, gp_stresses, cbar_stress,
            transform=False,
            use_z_coord=use_z_coord,
            use_rotations=use_rotations,
            use_gp_forces=use_gp_forces,
            use_axial_stress=use_axial_stress,
            use_mode_shapes_as_features=use_mode_shapes_as_features,
            prediction_type=prediction_type,
            virtual_edge_cache = VirtualEdgeCache(cache_file=os.path.join(data_dir, "virtual_edges_cache.pkl"))
        )
        
        node_features_trans, edge_index_trans, edge_features_trans, transformation_matrix, transform_info, transform, static_targets= create_graph_from_bdf(
            bdf_model, static_displacements, mode_shape, gp_forces, gp_stresses, cbar_stress,
            transform=True,
            use_z_coord=use_z_coord,
            use_rotations=use_rotations,
            use_gp_forces=use_gp_forces,
            use_axial_stress=use_axial_stress,
            use_mode_shapes_as_features=use_mode_shapes_as_features,
            prediction_type=prediction_type,
            virtual_edge_cache = VirtualEdgeCache(cache_file=os.path.join(data_dir, "virtual_edges_cache.pkl"))
        )
        
        # Convert tensors to numpy arrays for plotting
        node_features_orig_np = node_features_orig.numpy()
        node_features_trans_np = node_features_trans.numpy()
        edge_index_orig_np = edge_index_orig.numpy()
        edge_index_trans_np = edge_index_trans.numpy()
        edge_features_orig_np = edge_features_orig.numpy()
        edge_features_trans_np = edge_features_trans.numpy()
        
        # Create the comparison plot
        plt.figure(figsize=(20, 10))
        
        # Plot original graph
        plt.subplot(121)
        
        # Plot edges
        for i in range(edge_index_orig_np.shape[1]):
            start_idx = edge_index_orig_np[0, i]
            end_idx = edge_index_orig_np[1, i]
            start_pos = node_features_orig_np[start_idx, :2]
            end_pos = node_features_orig_np[end_idx, :2]
            
            # Color based on edge type
            if edge_features_orig_np[i, -1] == 1:  # Virtual edge
                color = 'green'
                alpha = 0.5
                linewidth = 1
            else:
                color = 'red' if edge_features_orig_np[i, 0] > 0.5 else 'yellow'
                alpha = 0.5
                linewidth = 2 if color == 'red' else 1
            
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                    color=color, alpha=alpha, linewidth=linewidth)
        
        # Plot nodes
        plt.scatter(node_features_orig_np[:, 0], node_features_orig_np[:, 1], 
                   c='blue', s=50)
        
        # Select nodes with stiffeners for highlighting
        stiffener_nodes = []
        for i in range(len(node_features_orig_np)):
            # Check stiffener bins (features 6-9)
            if np.any(node_features_orig_np[i, 6:10] > 0):
                stiffener_nodes.append(i)
        
        # Select a random node from nodes with stiffeners
        if stiffener_nodes:
            random_node = np.random.choice(stiffener_nodes)
        else:
            random_node = np.random.randint(0, len(node_features_orig_np))
            
        plt.scatter(node_features_orig_np[random_node, 0], node_features_orig_np[random_node, 1], 
                   c='red', s=100)
        
        # Add feature text for the selected node
        feature_names = [
            "X coord", "Y coord",           # 0-1: Coordinates
            "SPC1",                        # 2: SPC features
            "Force X", "Force Y",          # 3-4: External forces
            "Shell count",                 # 5: Shell element count
            "Stiff 0°/180°",              # 6: Stiffener bin 1
            "Stiff 45°/225°",             # 7: Stiffener bin 2
            "Stiff 90°/270°",             # 8: Stiffener bin 3
            "Stiff 135°/315°",            # 9: Stiffener bin 4
        ]

        if prediction_type == "buckling":
            feature_names.extend([
                "Displacement X",              # Static displacement X
                "Displacement Y",              # Static displacement Y
                "GP Stress σx",               # GP stress normal X
                "GP Stress σy",               # GP stress normal Y
                "GP Stress τxy"               # GP stress shear XY
            ])

        if use_gp_forces and prediction_type == "buckling":
            force_quadrant_names = [
                "GP Force Q1 X", "GP Force Q1 Y",  # Quadrant 1 forces
                "GP Force Q2 X", "GP Force Q2 Y",  # Quadrant 2 forces
                "GP Force Q3 X", "GP Force Q3 Y",  # Quadrant 3 forces
                "GP Force Q4 X", "GP Force Q4 Y"   # Quadrant 4 forces
            ]
            feature_names.extend(force_quadrant_names)
            
        if use_mode_shapes_as_features and prediction_type == "buckling":
            mode_shape_names = ["Mode Shape X", "Mode Shape Y", "Mode Shape Z"]
            feature_names.extend(mode_shape_names)
            if use_rotations:
                mode_shape_rot_names = ["Mode Shape θx", "Mode Shape θy", "Mode Shape θz"]
                feature_names.extend(mode_shape_rot_names)

        # Print total number of features for debugging
        print(f"\nTotal number of features in node_features: {node_features_orig_np.shape[1]}")
        print(f"Total number of feature names: {len(feature_names)}")
        
        # Ensure we have names for all features
        if node_features_orig_np.shape[1] > len(feature_names):
            print("\nWarning: Some features don't have names!")
            for i in range(len(feature_names), node_features_orig_np.shape[1]):
                feature_names.append(f"Unknown Feature {i}")
        
        feature_text = f"Node {random_node} Features (Original):\n"
        for i, (name, feat) in enumerate(zip(feature_names, node_features_orig_np[random_node])):
            feature_text += f"{name}: {feat:.4f}\n"
        plt.text(0.02, 0.98, feature_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title("Original Coordinates")
        plt.axis('equal')
        plt.grid(True)
        
        # Plot transformed graph
        plt.subplot(122)
        
        # Plot edges
        for i in range(edge_index_trans_np.shape[1]):
            start_idx = edge_index_trans_np[0, i]
            end_idx = edge_index_trans_np[1, i]
            start_pos = node_features_trans_np[start_idx, :2]
            end_pos = node_features_trans_np[end_idx, :2]
            
            # Color based on edge type
            if edge_features_trans_np[i, -1] == 1:  # Virtual edge
                color = 'green'
                alpha = 0.5
                linewidth = 1
            else:
                color = 'red' if edge_features_trans_np[i, 0] > 0.5 else 'yellow'
                alpha = 0.5
                linewidth = 2 if color == 'red' else 1
            
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                    color=color, alpha=alpha, linewidth=linewidth)
        
        # Plot nodes
        plt.scatter(node_features_trans_np[:, 0], node_features_trans_np[:, 1], 
                   c='blue', s=50)
        
        # Highlight the same node
        plt.scatter(node_features_trans_np[random_node, 0], node_features_trans_np[random_node, 1], 
                   c='red', s=100)
        
        # Add feature text for the selected node
        feature_text = f"Node {random_node} Features (Transformed):\n"
        for i, (name, feat) in enumerate(zip(feature_names, node_features_trans_np[random_node])):
            feature_text += f"{name}: {feat:.4f}\n"
        plt.text(0.02, 0.98, feature_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title("Transformed Coordinates")
        plt.axis('equal')
        plt.grid(True)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=2, label='Stiffener (CBAR)'),
            plt.Line2D([0], [0], color='yellow', label='Shell Element'),
            plt.Line2D([0], [0], color='green', label='Virtual Edge'),
            plt.Line2D([0], [0], marker='o', color='blue', label='Nodes',
                      markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='red', label='Selected Node',
                      markersize=10, linestyle='None')
        ]
        plt.figlegend(handles=legend_elements, loc='center right')
        
        plt.tight_layout()
        plt.savefig(f'graph_comparison_{file_idx}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Print detailed feature comparison
        print(f"\nDetailed Feature Comparison for Node {random_node}")
        print("\nFeature Name      Original         Transformed      Difference")
        print("-" * 60)
        for name, orig, trans in zip(feature_names, 
                                   node_features_orig_np[random_node], 
                                   node_features_trans_np[random_node]):
            diff = trans - orig
            print(f"{name:<15} {orig:15.4f} {trans:15.4f} {diff:15.4f}")

        # Additional information based on prediction type
        print(f"\nPrediction Type: {prediction_type}")
        if prediction_type == "buckling":
            print(f"Eigenvalue: {eigenvalue}")
        elif "static" in prediction_type:
            print("Static Analysis Features:")
            print(f"Number of displacement components: {static_displacements.shape[1]}")
            print(f"Number of stress components: {gp_stresses.shape[1]}")
        elif prediction_type == "mode_shape":
            print("Mode Shape Features:")
            print(f"Number of mode shape components: {mode_shape.shape[1]}")
            print(f"Associated eigenvalue: {eigenvalue}")

