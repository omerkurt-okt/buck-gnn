import os
import numpy as np
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import random
from typing import List, Dict, Tuple, Set, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import subprocess
from copy import deepcopy

class CustomBDF(BDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_header = """SOL 105
CEND
ECHO=NONE

SET 10 = 200
SUBCASE       1
  SUBTITLE  = Static
  LABEL= Static
  SPC =        1
  LOAD =        2
  DISPLACEMENT(PLOT) = ALL
  ELFORCE(PLOT) = ALL
  GPFORCE(PLOT) = ALL
  STRESS(PLOT) = ALL
  GPSTRESS(PLOT,PRINT) = 10
  STRAIN(PLOT) = ALL
STRFIELD = 10                                                                   
SUBCASE       2
  SUBTITLE  = Buckling
  LABEL= Buckling
  SPC =        1
  METHOD(STRUCTURE) =        1
  STATSUB(BUCKLING) =        1
  ANALYSIS = BUCK
  DISPLACEMENT(PLOT) = ALL
OUTPUT(POST)   
SET 2 = ALL
SURFACE 200 SET 2 NORMAL Z                                                      
BEGIN BULK
$PARAMS
PARAM       OMID     YES
PARAM       POST      -1
"""
    def _write_executive_control_deck(self, bdf_file):
        pass  # Skip default executive control deck

    def _write_case_control_deck(self, bdf_file):
        bdf_file.write(self.custom_header)

# Disable all logging
logging.disable(logging.CRITICAL)
# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('model_generation.log'),
#         logging.StreamHandler()
#     ]
# )

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

class LoadcaseType(Enum):
    COMPRESSION = "compression"
    COMPRESSION_SHEAR = "compression-shear"
    TENSION = "tension"
    TENSION_SHEAR = "tension-shear"
    SHEAR = "shear"
    MIXED = "mixed"

@dataclass
class Edge:
    node1: int
    node2: int
    direction: np.ndarray
    length: float
    element_id: Optional[int] = None

    def __hash__(self):
        return hash((min(self.node1, self.node2), max(self.node1, self.node2)))

    def __eq__(self, other):
        return (min(self.node1, self.node2), max(self.node1, self.node2)) == \
               (min(other.node1, other.node2), max(other.node1, other.node2))

@dataclass
class LoadCase:
    bcs: List[Tuple[int, str]]  # (node_id, dof)
    loads: List[Tuple[int, np.ndarray, float]]  # (node_id, direction, magnitude)
    type: LoadcaseType = None
    eigenvalue_ratio: float = None
    stress_data: Dict = None

class ModelGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.next_element_id = 100000
        os.makedirs(self.config.temp_dir, exist_ok=True)

    def detect_boundary(self, model: BDF) -> Set[int]:
        """Detect only outer boundary nodes using rightmost node method"""
        # Find all boundary edges (appearing once)
        edge_count = defaultdict(int)
        node_coords = {nid: np.array(node.xyz[:2]) for nid, node in model.nodes.items()}
        
        # Count edge occurrences
        for elem in model.elements.values():
            if elem.type in ['CQUAD4', 'CTRIA3']:
                nodes = elem.nodes
                for i in range(len(nodes)):
                    n1, n2 = nodes[i], nodes[(i+1) % len(nodes)]
                    edge = tuple(sorted([n1, n2]))
                    edge_count[edge] += 1

        # Get all boundary edges
        boundary_edges = set((n1, n2) for (n1, n2), count in edge_count.items() if count == 1)
        
        # Find rightmost node (definitely on outer boundary)
        rightmost_node = max(node_coords.keys(), key=lambda nid: node_coords[nid][0])
        
        # Trace outer boundary starting from rightmost node
        outer_boundary = set([rightmost_node])
        current = rightmost_node
        
        while True:
            # Find next connected boundary edge
            next_edge = None
            for edge in boundary_edges:
                if edge[0] == current:
                    next_edge = edge
                    break
                elif edge[1] == current:
                    next_edge = (edge[1], edge[0])  # Reverse edge direction
                    break
                    
            if next_edge is None or next_edge[1] == rightmost_node:
                break
                
            current = next_edge[1]
            outer_boundary.add(current)
            boundary_edges.remove(tuple(sorted(next_edge)))
        
        return outer_boundary

    def find_connected_boundary_nodes(self, node: int, boundary_nodes: Set[int], model: BDF) -> List[int]:
        """Find consecutive boundary nodes connected to given node"""
        connected = []
        current = node
        nodes_coords = {nid: np.array(model.nodes[nid].xyz[:2]) for nid in boundary_nodes}
        visited = {node}
        
        while True:
            neighbors = []
            for elem in model.elements.values():
                if elem.type in ['CQUAD4', 'CTRIA3']:
                    nodes = elem.nodes
                    for i, n1 in enumerate(nodes):
                        if n1 == current:
                            n2 = nodes[(i+1) % len(nodes)]
                            if n2 in boundary_nodes and n2 not in visited:
                                neighbors.append(n2)
                            n2 = nodes[(i-1) % len(nodes)]
                            if n2 in boundary_nodes and n2 not in visited:
                                neighbors.append(n2)
            
            if not neighbors:
                break
                
            # Choose the closest neighbor
            current_pos = nodes_coords[current]
            next_node = min(neighbors, 
                           key=lambda n: np.linalg.norm(nodes_coords[n] - current_pos))
            
            connected.append(next_node)
            visited.add(next_node)
            current = next_node
            
        return connected

    def create_edges(self, model: BDF) -> Set[Edge]:
        """Create edges from 2D elements including diagonals for quads"""
        edges = set()
        
        for elem in model.elements.values():
            if elem.type in ['CQUAD4', 'CTRIA3']:
                nodes = elem.nodes
                # Add perimeter edges
                for i in range(len(nodes)):
                    n1, n2 = nodes[i], nodes[(i+1) % len(nodes)]
                    node1_coords = np.array(model.nodes[n1].xyz[:2])
                    node2_coords = np.array(model.nodes[n2].xyz[:2])
                    direction = node2_coords - node1_coords
                    length = np.linalg.norm(direction)
                    direction = direction / length
                    edges.add(Edge(n1, n2, direction, length))

                # Add diagonals for quads
                if elem.type == 'CQUAD4':
                    for i, j in [(0, 2), (1, 3)]:
                        n1, n2 = nodes[i], nodes[j]
                        node1_coords = np.array(model.nodes[n1].xyz[:2])
                        node2_coords = np.array(model.nodes[n2].xyz[:2])
                        direction = node2_coords - node1_coords
                        length = np.linalg.norm(direction)
                        direction = direction / length
                        edges.add(Edge(n1, n2, direction, length))

        return edges

    def create_stiffener_properties(self, model: BDF) -> None:
        """Create material and property cards for stiffeners"""
        # Create material
        if 4 not in model.materials:
            model.add_mat1(mid=4, E=76000.,G=None, nu=0.3, a=None)

        # Create property for inactive stiffeners (dummy)
        if 999 not in model.properties:
            model.add_pbar(999, 4, 0.0001, 0.0001, 0.0001,0, 0.000001)

        # Create property for active stiffeners
        if 900 not in model.properties:
            area = 2.0 * 80.0  # mm²
            i1 = (2.0 * 80.0**3) / 12.0  # mm⁴
            i2 = (80.0 * 2.0**3) / 12.0  # mm⁴
            j = 0.333 * 80.0 * 2**3  # mm⁴
            model.add_pbar(900, 4, area, i1, i2, 0, j)

    def create_stiffener_elements(self, model: BDF, edges: Set[Edge]) -> None:
        """Create CBAR elements for all edges"""
        for edge in edges:
            if edge.element_id is None:
                edge.element_id = self.next_element_id
                self.next_element_id += 1
                model.add_cbar(edge.element_id, 999, [edge.node1, edge.node2], x=[0.0, 0.0, 1.0], g0=None)

    def are_directions_similar(self, dir1: np.ndarray, dir2: np.ndarray) -> Tuple[bool, float]:
        """Check if two directions are within tolerance and return the angle between them
        
        Returns:
            Tuple of (is_similar: bool, angle_deg: float)
        """
        angle = np.arccos(np.clip(np.abs(np.dot(dir1, dir2)), -1.0, 1.0))
        angle_deg = np.degrees(angle)
        return angle_deg <= self.config.direction_tolerance, angle_deg

    def find_connected_edges(self, edge: Edge, available_edges: Set[Edge], 
                            check_node1: bool = True, check_node2: bool = True) -> List[Edge]:
        """Find edges connected to specified nodes of the given edge and having similar direction."""
        connected = []
        for other in available_edges:
            if edge == other:
                continue
                
            # Check connections to node2 (forward direction)
            if check_node2:
                if edge.node2 == other.node1:
                    is_similar, angle = self.are_directions_similar(edge.direction, other.direction)
                    if is_similar:
                        connected.append((other, angle))
                elif edge.node2 == other.node2:
                    is_similar, angle = self.are_directions_similar(edge.direction, -other.direction)
                    if is_similar:
                        # Modify direction in place
                        other.node1, other.node2 = other.node2, other.node1
                        other.direction = -other.direction
                        connected.append((other, angle))
                    
            # Check connections to node1 (backward direction)
            if check_node1:
                if edge.node1 == other.node2:
                    is_similar, angle = self.are_directions_similar(edge.direction, other.direction)
                    if is_similar:
                        # Modify direction in place
                        other.node1, other.node2 = other.node2, other.node1
                        other.direction = -other.direction
                        connected.append((other, angle))
                elif edge.node1 == other.node1:
                    is_similar, angle = self.are_directions_similar(edge.direction, -other.direction)
                    if is_similar:
                        connected.append((other, angle))
        
        # Sort by angle (most aligned first) and return only the edges
        connected.sort(key=lambda x: x[1])
        return [edge for edge, angle in connected]

    def activate_stiffener_group(self, edges: Set[Edge], model: BDF) -> None:
        """Activate a group of connected stiffeners"""
        available_edges = edges.copy()
        num_to_activate = random.randint(self.config.min_active_stiffeners, 
                                    min(self.config.max_active_stiffeners, len(edges)))
        activated = set()

        while len(activated) < num_to_activate and available_edges:
            min_group = min(self.config.min_consecutive, num_to_activate - len(activated))
            max_group = min(self.config.max_consecutive, num_to_activate - len(activated))
            group_size = random.randint(min_group, max_group)
            
            start_edge = random.choice(list(available_edges))
            current_group = [start_edge]
            available_edges.remove(start_edge)
            
            forward_growing = True
            backward_tried = False
            
            while len(current_group) < group_size and available_edges:
                if forward_growing:
                    connected = self.find_connected_edges(current_group[-1], available_edges, check_node1=False)
                    
                    if not connected and not backward_tried:
                        forward_growing = False
                        backward_tried = True
                        continue
                    elif not connected:
                        break
                else:
                    connected = self.find_connected_edges(current_group[0], available_edges, check_node2=False)
                    if not connected:
                        break
                    
                next_edge = connected[0]
                if forward_growing:
                    current_group.append(next_edge)
                else:
                    current_group.insert(0, next_edge)
                available_edges.remove(next_edge)
            
            activated.update(current_group)

        # Update element properties in the model
        for edge in edges:
            pid = 900 if edge in activated else 999
            model.elements[edge.element_id].pid = pid

    def generate_loadcase(self, model: BDF) -> Optional[LoadCase]:
        """Generate a load case with boundary conditions and forces"""
        boundary_nodes = self.detect_boundary(model)
        
        self.logger.info(f"Boundary nodes: {len(boundary_nodes)}, Required min nodes: {self.config.min_nodes_per_line}")
        if len(boundary_nodes) < self.config.min_nodes_per_line * 2:
            self.logger.info("Not enough boundary nodes for minimum line length")
            return None
            
        # Apply BCs
        bcs = []
        bc_nodes = set()
        for _ in range(self.config.max_bc_lines):
            num_nodes = random.randint(self.config.min_nodes_per_line, 
                                    self.config.max_nodes_per_line)
            
            # Get connected nodes
            start_node = random.choice(list(boundary_nodes - bc_nodes))
            connected = self.find_connected_boundary_nodes(
                start_node,
                boundary_nodes - bc_nodes,
                model
            )
            
            # Check if we have enough connected nodes
            if len(connected) + 1 < num_nodes:  # +1 for start_node
                continue
                
            # Select exact number of nodes
            selected = [start_node] + connected[:num_nodes-1]
            if len(selected) == num_nodes:  # Only proceed if we got exactly what we needed
                bc_nodes.update(selected)
                for node in selected:
                    bcs.append((node, '123456'))
        
        # Apply loads
        loads = []
        available_nodes = boundary_nodes - bc_nodes
        num_load_lines = self.config.max_load_lines
        
        for _ in range(num_load_lines):
            if len(available_nodes) < self.config.min_nodes_per_load_line:
                break
                
            num_nodes = random.randint(self.config.min_nodes_per_load_line, 
                                    min(self.config.max_nodes_per_load_line, len(available_nodes)))
            
            # Get connected nodes
            start_node = random.choice(list(available_nodes))
            connected = self.find_connected_boundary_nodes(
                start_node,
                available_nodes,
                model
            )
            
            # Check if we have enough connected nodes
            if len(connected) + 1 < num_nodes:  # +1 for start_node
                continue
                
            # Select exact number of nodes
            selected = [start_node] + connected[:num_nodes-1]
            if len(selected) == num_nodes:  # Only proceed if we got exactly what we needed
                available_nodes -= set(selected)
                
                # Generate random load
                angle = random.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle), 0])
                magnitude = random.uniform(self.config.min_load, self.config.max_load)
                
                for node in selected:
                    loads.append((node, direction, magnitude))
        
        return LoadCase(bcs=bcs, loads=loads) if bcs and loads else None
    
    def create_analysis_model(self, base_model: BDF, loadcase: LoadCase, 
                            num_eigenvalues: int, edges: Set[Edge] = None,
                            generate_stiffeners: bool = False) -> BDF:
        """Create analysis model with specified number of eigenvalues"""
        model = CustomBDF(debug=None)
        model.read_bdf(base_model.bdf_filename)
        
        # Apply BCs and loads
        for node_id, dof in loadcase.bcs:
            model.add_spc1(1, dof, [node_id])
        for node_id, direction, magnitude in loadcase.loads:
            model.add_force(2, node_id, magnitude, direction)
            
        # Generate stiffeners with dummy props
        if edges:
            self.create_stiffener_properties(model)
            self.create_stiffener_elements(model, edges)

        # Add stiffeners if requested
        if generate_stiffeners and edges:
            self.activate_stiffener_group(edges, model)

        # Add EIGRL card
        if num_eigenvalues>2:
            model.add_eigrl(1, nd=num_eigenvalues)
        else:
            model.add_eigrl(1, 0.0,nd=num_eigenvalues)
        return model

    def run_nastran(self, bdf_path: str) -> Optional[str]:
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
                self.config.nastran_path,
                os.path.basename(local_bdf),  # Use basename since we're changing directory
                "scr=yes",
                "mem=2048"
            ]
            
            process = subprocess.run(
                cmd,
                cwd=working_dir,  # Set working directory for Nastran
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if process.returncode != 0:
                self.logger.error(f"Nastran analysis failed: {process.stderr}")
                return None
            
            # Return the full path to the op2 file
            op2_path = os.path.join(working_dir, os.path.basename(op2_file))
            return op2_path if os.path.exists(op2_path) else None
            
        except Exception as e:
            self.logger.error(f"Error running Nastran: {str(e)}")
            return None

    
    def analyze_model(self, model: BDF, temp_path: str) -> Tuple[bool, Optional[float], Optional[Dict]]:
        """Analyze a model and get results"""
        try:
            # Write model to temporary file
            model.write_bdf(temp_path)
            
            # Run Nastran
            op2_file = self.run_nastran(temp_path)
            if not op2_file:
                return False, None, None

            # Parse results
            op2 = OP2()
            op2.read_op2(op2_file)
            
            # Get eigenvalues
            eigenvalue_ratio = None
            all_buck_subcases = list(op2.eigenvectors.keys())
            buck_isubcase = all_buck_subcases[0]
            if buck_isubcase in op2.eigenvectors:
                eigenvalues = op2.eigenvectors[buck_isubcase].eigrs
                first_eigenvalue = eigenvalues[0]
                first_positive = next((ev for ev in eigenvalues if ev > 0), None)
                
                if first_positive:
                    eigenvalue_ratio = abs(first_positive / first_eigenvalue)

            def make_unique_groups(input_array):

                if input_array.shape[0] % 3 != 0:
                    raise ValueError("Number of rows must be a multiple of 3")

                # Reshape the array into groups of 3 rows
                grouped = input_array.reshape(-1, 3, input_array.shape[1])

                # Convert groups to a structured array for easy comparison
                dtype = [('row1', grouped.dtype, grouped.shape[2]),
                        ('row2', grouped.dtype, grouped.shape[2]),
                        ('row3', grouped.dtype, grouped.shape[2])]
                structured = np.array([tuple(group) for group in grouped], dtype=dtype)

                # Find unique groups
                unique_groups, indices = np.unique(structured, return_index=True)

                # Sort indices to maintain original order
                indices.sort()

                # Reconstruct the array from unique groups
                for indice in indices:
                    first_elements = [grouped[indice][0] for indice in indices]
                first_elements_array = np.array(first_elements)

                return first_elements_array
            # Get stresses
            stress_data = None
            all_gpstress_subcases = list(op2.grid_point_surface_stresses.keys())
            gpstress_isubcase = all_gpstress_subcases[0]
            if gpstress_isubcase in op2.grid_point_surface_stresses:
                stresses = op2.grid_point_surface_stresses[gpstress_isubcase].data[0]
                stresses = make_unique_groups(stresses)
                minor = np.mean(stresses[:, 5])
                major = np.mean(stresses[:, 4])
                
                denominator = major - minor
                if denominator > 0:
                    compression_ratio = minor / denominator
                    compression_ratio = min(max(compression_ratio, -1), 0)
                    
                    tension_ratio = major / denominator
                    tension_ratio = min(max(tension_ratio, 0), 1)
                    
                    stress_data = {
                        'compression_ratio': compression_ratio,
                        'tension_ratio': tension_ratio
                    }

            # Clean up
            if self.config.delete_temp_files:
                for ext in ['.log', '.f04', '.f06', '.op2', '.bdf']:
                    file_to_remove = os.path.splitext(temp_path)[0] + ext
                    if os.path.exists(file_to_remove):
                        os.remove(file_to_remove)
            return True, eigenvalue_ratio, stress_data

        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}")
            return False, None, None
        
    def determine_loadcase_type(self, stress_data: Dict) -> LoadcaseType:
        """Determine the type of load case based on stress ratios"""
        if not stress_data:
            return LoadcaseType.MIXED
            
        compression_ratio = stress_data['compression_ratio']
        tension_ratio = stress_data['tension_ratio']
        
        if compression_ratio <= -0.8:
            return LoadcaseType.COMPRESSION
        elif compression_ratio <= -0.65:
            return LoadcaseType.COMPRESSION_SHEAR
        elif tension_ratio >= 0.8:
            return LoadcaseType.TENSION
        elif tension_ratio >= 0.65:
            return LoadcaseType.TENSION_SHEAR
        elif max(abs(compression_ratio), tension_ratio) < 0.55:
            return LoadcaseType.SHEAR
        else:
            return LoadcaseType.MIXED
    def should_accept_loadcase(self, loadcase_type: LoadcaseType, eigenvalue_ratio: float) -> bool:
        """Determine if a load case should be accepted"""
        if eigenvalue_ratio is None:
            self.logger.info("Rejecting loadcase due to None eigenvalue ratio")
            return False

        if eigenvalue_ratio <= self.config.eigenvalue_ratio_limit:
            self.logger.info(f"Accepting loadcase with eigenvalue ratio {eigenvalue_ratio} (below limit)")
            return True

        if eigenvalue_ratio <= 10:
            accept = random.random() < self.config.high_ratio_acceptance_rate
            self.logger.info(f"Loadcase with ratio {eigenvalue_ratio} <= 10: {'accepted' if accept else 'rejected'}")
            return accept
        
        if loadcase_type == LoadcaseType.TENSION or loadcase_type == LoadcaseType.TENSION_SHEAR:
            accept = random.random() < self.config.very_high_ratio_acceptance_rate
            self.logger.info(f"Loadcase with ratio {eigenvalue_ratio} > 10: {'accepted' if accept else 'rejected'}")
        else:
            accept = random.random() < self.config.high_ratio_acceptance_rate
            self.logger.info(f"Loadcase with ratio {eigenvalue_ratio} <= 10: {'accepted' if accept else 'rejected'}")

        return accept
    
    def process_model(self, bdf_path: str, output_base_dir: str) -> None:
        """Process a single model file"""
        try:
            self.logger.info(f"Processing model: {bdf_path}")
            
            # Read the model
            model = BDF(debug=None)

            model.read_bdf(bdf_path)
            
            # Create edges if generating stiffeners
            # edges = self.create_edges(model) if self.config.generate_stiffeners else None
            
            original_edges = self.create_edges(model)
            # Create output directories
            output_dirs = {lt: os.path.join(output_base_dir, lt.value) for lt in LoadcaseType}
            for directory in output_dirs.values():
                os.makedirs(directory, exist_ok=True)
            
            successful_loadcases = []
            trials = 0
            
            # Generate and test load cases
            while len(successful_loadcases) < self.config.loadcases_per_model and trials < self.config.max_trials:
                
                loadcase = self.generate_loadcase(model)
                if not loadcase:
                    self.logger.info("Failed to generate valid loadcase")
                    trials += 1
                    continue

                self.logger.info(f"Generated loadcase with {len(loadcase.bcs)} BCs and {len(loadcase.loads)} loads")

                # Test with 50 eigenvalues
                test_model = self.create_analysis_model(model, loadcase, num_eigenvalues=50)
                temp_path = os.path.join(self.config.temp_dir, f"test_{hash(str(loadcase))}.bdf")
                
                success, eigenvalue_ratio, stress_data = self.analyze_model(test_model, temp_path)
                
                if not success:
                    trials += 1
                    continue
                self.logger.info(f"Analysis results - eigenvalue ratio: {eigenvalue_ratio}, stress data: {stress_data}")
                # Determine load case type and check acceptance
                loadcase_type = self.determine_loadcase_type(stress_data)
                loadcase.type = loadcase_type
                loadcase.eigenvalue_ratio = eigenvalue_ratio
                loadcase.stress_data = stress_data
                
                if self.should_accept_loadcase(loadcase_type, eigenvalue_ratio):
                    self.logger.info(f"Accepted loadcase of type {loadcase_type}")
                    successful_loadcases.append(loadcase)
                    trials = 0
                else:
                    self.logger.info(f"Rejected loadcase of type {loadcase_type}")
                    trials += 1
            
            # Process accepted load cases
            for i, loadcase in enumerate(successful_loadcases):
                # Create directory for this load case
                base_name = f"{os.path.splitext(os.path.basename(bdf_path))[0]}_lc{i}"

                # loadcase_dir = os.path.join(output_dirs[loadcase.type], base_name)
                # os.makedirs(loadcase_dir, exist_ok=True)

                loadcase_dir = output_dirs[loadcase.type]
                
                # Create and save pristine model with 1 eigenvalue
                # if not self.config.generate_stiffeners:
                pristine_edges = deepcopy(original_edges)
                pristine_model = self.create_analysis_model(model, loadcase, num_eigenvalues=1, edges=pristine_edges, generate_stiffeners=False)
                pristine_path = os.path.join(loadcase_dir, f"{base_name}_pristine.bdf")
                pristine_model.cross_reference(pristine_model._xref)
                pristine_model.write_bdf(pristine_path)
                
                # Generate stiffener patterns if requested
                if self.config.generate_stiffeners and original_edges:
                    for j in range(self.config.patterns_per_loadcase):
                        pattern_edges = deepcopy(original_edges)
                        pattern_model = self.create_analysis_model(
                            model, loadcase, num_eigenvalues=1,
                            edges=pattern_edges, generate_stiffeners=True
                        )
                        pattern_path = os.path.join(loadcase_dir, f"{base_name}_pattern{j}.bdf")
                        pattern_model.cross_reference(pattern_model._xref)
                        pattern_model.write_bdf(pattern_path)
                
            self.logger.info(f"Successfully processed {len(successful_loadcases)} load cases for {bdf_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing {bdf_path}: {str(e)}")
            raise

def process_model_wrapper(args):
    """Wrapper function for multiprocessing"""
    bdf_path, output_dir, config = args
    try:
        generator = ModelGenerator(config)
        generator.process_model(bdf_path, output_dir)
        return True, bdf_path
    except Exception as e:
        return False, f"Failed to process {bdf_path}: {str(e)}"

def process_directory(input_dir: str, output_dir: str, config: Config, num_processes: int = None) -> None:
    """Process all BDF files in a directory using multiprocessing"""
    if num_processes is None:
        num_processes = os.cpu_count()  # Use all available CPU cores

    bdf_files = [f for f in os.listdir(input_dir) if f.endswith('.bdf')]
    total_files = len(bdf_files)
    
    # Create process-specific configs with unique temp directories
    process_configs = []
    for i in range(num_processes):
        process_config = deepcopy(config)
        process_config.temp_dir = os.path.join(input_dir, f"temp_analysis_process_{i}")
        os.makedirs(process_config.temp_dir, exist_ok=True)
        process_configs.append(process_config)
    
    # Distribute files among processes
    process_args = []
    for i, bdf_file in enumerate(bdf_files):
        config_index = i % num_processes
        process_args.append((
            os.path.join(input_dir, bdf_file),
            output_dir,
            process_configs[config_index]
        ))

    # Process files using multiprocessing
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(
            executor.map(process_model_wrapper, process_args),
            total=total_files,
            desc=f"Processing models using {num_processes} processes"
        ))

    # Clean up temporary directories
    for i in range(num_processes):
        process_temp_dir = os.path.join(input_dir, f"temp_analysis_process_{i}")
        try:
            shutil.rmtree(process_temp_dir)
        except Exception as e:
            logging.error(f"Failed to remove temporary directory {process_temp_dir}: {str(e)}")

    # Report results
    successful = sum(1 for success, _ in results if success)
    print(f"\nProcessing complete: {successful}/{total_files} files processed successfully")
    
    # Print errors if any
    errors = [msg for success, msg in results if not success]
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)

def main():
    input_dir = r"D:\Projects_Omer\GNN_Project\0_Data\TEST\SHAPES\Shapes_50_r\w_cutout"
    output_dir = r"D:\Projects_Omer\GNN_Project\0_Data\TEST\w_stiffener\Shapes_50_r"
    from time import sleep
    sleep(4300)
    # Configuration
    config = Config(
        min_load=10.0,
        max_load=1000.0,
        generate_stiffeners=True,
        min_active_stiffeners=10,
        max_active_stiffeners=100,
        min_consecutive=5,
        max_consecutive=25,
        loadcases_per_model=4,
        patterns_per_loadcase=5,
        max_bc_lines=1,
        max_load_lines=1,
        max_nodes_per_line=35,
        min_nodes_per_line=25,
        max_nodes_per_load_line=20,
        min_nodes_per_load_line=10,
        direction_tolerance=30,
        max_trials=100,
        eigenvalue_ratio_limit=10,
        high_ratio_acceptance_rate=0.3,
        very_high_ratio_acceptance_rate=0.1,
        nastran_path=r"C:\Program Files\MSC.Software\MSC_Nastran\2020sp1\bin\nastran.exe",
        temp_dir="temp_analysis",
        delete_temp_files=True
    )

    # Use 50% of available CPU cores by default
    num_processes = max(1, int(os.cpu_count() * 0.50))
    process_directory(input_dir, output_dir, config, num_processes)

if __name__ == "__main__":
    main()