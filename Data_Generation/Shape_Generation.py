import numpy as np
import random
from scipy.special import comb
import os
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax2
from OCC.Core.Geom import Geom_BezierCurve
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,
                                    BRepBuilderAPI_MakeFace)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.GC import GC_MakeEllipse, GC_MakeCircle
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

class OrganicShapeGenerator:
    def __init__(self, config):
        self.config = config
        os.makedirs(config['output_dir'], exist_ok=True)

    def generate_natural_boundary(self):
        """Generate boundary points for organic-asymmetric shapes"""
        num_points = random.randint(
            self.config['boundary']['min_points'],
            self.config['boundary']['max_points']
        )
        
        points = []
        # Base radius in millimeters
        base_radius = random.uniform(
            self.config['boundary']['min_radius'],
            self.config['boundary']['max_radius']
        )
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            
            angle += random.uniform(
                -self.config['boundary']['angle_variation'],
                self.config['boundary']['angle_variation']
            )
            
            radius = base_radius * (1 + random.uniform(
                self.config['boundary']['min_radius_variation'],
                self.config['boundary']['max_radius_variation']
            ))
            
            freq_multiplier = self.config['boundary']['frequency_multiplier']
            freq_magnitude = self.config['boundary']['frequency_magnitude']
            radius *= (1 + freq_magnitude * 
                      np.sin(freq_multiplier * angle + random.uniform(-np.pi, np.pi)))
            
            if random.random() < self.config['boundary']['inward_curve_probability']:
                radius *= random.uniform(
                    self.config['boundary']['min_inward_scale'],
                    self.config['boundary']['max_inward_scale']
                )
            
            point = np.array([np.cos(angle), np.sin(angle)]) * radius
            points.append(point)
        
        return np.array(points)

    def generate_smooth_control_points(self, p1, p2, prev_point=None, next_point=None):
        """Generate control points with consistent minimum radius for all transitions"""
        base_dir = p2 - p1
        length = np.linalg.norm(base_dir)
        
        # Define minimum radius for all transitions
        min_radius = length * self.config['curvature']['min_radius_factor']
        
        if prev_point is not None and next_point is not None:
            prev_dir = p1 - prev_point
            next_dir = next_point - p2
            
            prev_dir_norm = prev_dir / np.linalg.norm(prev_dir)
            next_dir_norm = next_dir / np.linalg.norm(next_dir)
            base_dir_norm = base_dir / length
            
            angle_in = np.arccos(np.clip(np.dot(prev_dir_norm, base_dir_norm), -1.0, 1.0))
            angle_out = np.arccos(np.clip(np.dot(base_dir_norm, next_dir_norm), -1.0, 1.0))
            
            def get_control_length(angle):
                base_length = min_radius * (4/3)
                variation_factor = random.uniform(
                    1.0,
                    1.0 + self.config['curvature']['length_variation']
                )
                return base_length * variation_factor
            
            entry_length = get_control_length(angle_in)
            exit_length = get_control_length(angle_out)
            
            entry_dir = (prev_dir_norm + base_dir_norm)
            exit_dir = (base_dir_norm + next_dir_norm)
            
            entry_dir = entry_dir / np.linalg.norm(entry_dir) * entry_length
            exit_dir = exit_dir / np.linalg.norm(exit_dir) * exit_length
            
            perpendicular = np.array([-base_dir[1], base_dir[0]]) / length
            max_variation = min_radius * self.config['curvature']['max_variation_scale']
            
            variation = random.uniform(-max_variation, max_variation)
            entry_dir += perpendicular * variation
            exit_dir += perpendicular * variation
            
        else:
            control_length = min_radius * (4/3) * random.uniform(
                1.0,
                1.0 + self.config['curvature']['length_variation']
            )
            entry_dir = exit_dir = base_dir / length * control_length
        
        c1 = p1 + entry_dir
        c2 = p2 - exit_dir
        
        return np.array([p1, c1, c2, p2])
    
    def scale_shape_to_bounds(self, points):
        """Scale points to fit within maximum dimension while maintaining aspect ratio"""
        # Get current bounds
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        current_width = np.max(x_coords) - np.min(x_coords)
        current_height = np.max(y_coords) - np.min(y_coords)
        
        # Get current maximum dimension
        current_max_dim = max(current_width, current_height)
        
        # Scale to fit within maximum size bound
        target_size = random.uniform(
            self.config['size_range']['min'],
            self.config['size_range']['max']
        )
        scale_factor = target_size / current_max_dim
        points = points * scale_factor
        
        # Center the shape at origin
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        center_x = (np.max(x_coords) + np.min(x_coords)) / 2
        center_y = (np.max(y_coords) + np.min(y_coords)) / 2
        
        points[:, 0] -= center_x
        points[:, 1] -= center_y
        
        return points
    
    def verify_shape_proportions(self, points):
        """Verify that shape meets aspect ratio requirements"""
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        
        aspect_ratio = width / height
        
        return (self.config['size_range']['aspect_ratio_min'] <= aspect_ratio <= 
                self.config['size_range']['aspect_ratio_max'])

    def point_segment_distance(self, p, a, b):
        """Calculate minimum distance from point p to line segment ab"""
        ab = b - a
        ap = p - a
        
        t = np.dot(ap, ab) / np.dot(ab, ab)
        
        if t < 0:
            return np.linalg.norm(p - a)
        elif t > 1:
            return np.linalg.norm(p - b)
        else:
            projection = a + t * ab
            return np.linalg.norm(p - projection)

    def is_point_inside_shape(self, point, boundary_points):
        """Check if a point is inside the shape using ray casting"""
        n = len(boundary_points)
        inside = False
        
        for i in range(n):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % n]
            
            if ((p1[1] > point[1]) != (p2[1] > point[1]) and
                point[0] < (p2[0] - p1[0]) * (point[1] - p1[1]) / 
                (p2[1] - p1[1]) + p1[0]):
                inside = not inside
                
        return inside

    def minimum_distance_to_boundary(self, point, boundary_points):
        """Calculate minimum distance from point to any boundary segment"""
        min_dist = float('inf')
        
        for i in range(len(boundary_points)):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % len(boundary_points)]
            
            dist = self.point_segment_distance(point, p1, p2)
            min_dist = min(min_dist, dist)
        
        return min_dist

    def calculate_shape_interior(self, boundary_points, min_distance):
        """Calculate the safe interior region for cutout placement"""
        x_coords = [p[0] for p in boundary_points]
        y_coords = [p[1] for p in boundary_points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        grid_size = min_distance / 2
        x_grid = np.arange(min_x, max_x, grid_size)
        y_grid = np.arange(min_y, max_y, grid_size)
        
        interior_points = []
        
        for x in x_grid:
            for y in y_grid:
                point = np.array([x, y])
                
                if self.is_point_inside_shape(point, boundary_points) and \
                   self.minimum_distance_to_boundary(point, boundary_points) >= min_distance:
                    interior_points.append(point)
        
        return np.array(interior_points) if interior_points else np.array([])
    
    def distribute_cutouts(self, face, boundary_points, config):
        """Distribute cutouts within the shape"""
        min_distance = config['cutout']['min_size'] * \
                      (1 + config['cutout']['min_distance_factor'])
        
        # Get safe interior points
        interior_points = self.calculate_shape_interior(boundary_points, min_distance)
        
        if len(interior_points) == 0:
            return None, 0
        
        # Determine number of cutouts based on shape area
        props = GProp_GProps()
        brepgprop_SurfaceProperties(face, props)
        shape_area = props.Mass()
        
        # Calculate maximum possible cutouts based on area and minimum spacing
        max_cutouts = int(shape_area / (np.pi * (min_distance/2)**2))
        desired_cutouts = random.randint(1, min(max_cutouts, 
                                              config['cutout']['max_cutouts']))
        
        cutouts_placed = 0
        attempts = 0
        max_attempts = config['cutout']['max_attempts']
        placed_centers = []
        
        while cutouts_placed < desired_cutouts and attempts < max_attempts:
            if len(interior_points) == 0:
                break
                
            point_idx = random.randint(0, len(interior_points) - 1)
            center = interior_points[point_idx]
            
            valid_position = True
            for placed_center in placed_centers:
                if np.linalg.norm(center - placed_center) < min_distance:
                    valid_position = False
                    break
            
            if not valid_position:
                attempts += 1
                continue
            
            available_space = self.minimum_distance_to_boundary(center, boundary_points)
            max_size = min(
                available_space * 2 / (1 + config['cutout']['min_distance_factor']),
                config['cutout']['max_size']
            )
            
            if max_size < config['cutout']['min_size']:
                attempts += 1
                continue
            
            cutout_size = random.uniform(
                config['cutout']['min_size'],
                max_size
            )
            
            try:
                ax2 = gp_Ax2(gp_Pnt(center[0], center[1], 0), gp_Dir(0, 0, 1))
                if random.random() < config['cutout']['circle_probability']:
                    curve = GC_MakeCircle(ax2, cutout_size/2).Value()
                else:
                    major_axis = cutout_size * random.uniform(
                        config['cutout']['ellipse_aspect_min'],
                        config['cutout']['ellipse_aspect_max']
                    )
                    curve = GC_MakeEllipse(ax2, major_axis/2, cutout_size/2).Value()
                
                edge = BRepBuilderAPI_MakeEdge(curve).Edge()
                wire = BRepBuilderAPI_MakeWire(edge).Wire()
                cutout_face = BRepBuilderAPI_MakeFace(wire).Face()
                face = BRepAlgoAPI_Cut(face, cutout_face).Shape()
                placed_centers.append(center)
                cutouts_placed += 1
                
                mask = [np.linalg.norm(p - center) >= min_distance for p in interior_points]
                interior_points = interior_points[mask]
                
            except Exception as e:
                attempts += 1
                continue
        
        if cutouts_placed > 0:
            return face, cutouts_placed
        return None, 0

    def create_shape(self, shape_index):
        max_attempts = self.config['generation']['max_attempts']
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Generate boundary points
                boundary_points = self.generate_natural_boundary()
                scaled_points = self.scale_shape_to_bounds(boundary_points)
                
                # Check if shape meets aspect ratio requirements
                if not self.verify_shape_proportions(scaled_points):
                    attempt += 1
                    continue
                
                # Rest of the shape creation code remains the same
                edges = []
                num_points = len(scaled_points)
                
                for i in range(num_points):
                    current = scaled_points[i]
                    next_point = scaled_points[(i + 1) % num_points]
                    prev_point = scaled_points[i - 1]
                    next_next = scaled_points[(i + 2) % num_points]
                    
                    control_points = self.generate_smooth_control_points(
                        current, next_point, prev_point, next_next
                    )
                    
                    bezier_curve = self.create_bezier_curve(control_points)
                    edge = BRepBuilderAPI_MakeEdge(bezier_curve).Edge()
                    edges.append(edge)
                
                wire_maker = BRepBuilderAPI_MakeWire()
                for edge in edges:
                    wire_maker.Add(edge)
                wire = wire_maker.Wire()
                face = BRepBuilderAPI_MakeFace(wire).Face()
                
                if self.config['with_cutouts']:
                    face, num_cutouts = self.distribute_cutouts(face, scaled_points, self.config)
                    if face is None:
                        attempt += 1
                        continue
                
                filename = f"{self.config['file_prefix']}_{shape_index:04d}.stp"
                writer = STEPControl_Writer()
                writer.Transfer(face, STEPControl_AsIs)
                status = writer.Write(os.path.join(self.config['output_dir'], filename))
                
                return status == IFSelect_RetDone
                
            except Exception as e:
                print(f"Attempt {attempt} failed: {str(e)}")
                attempt += 1
                
        return False

    def create_bezier_curve(self, control_points):
        """Create OpenCASCADE Bezier curve from control points"""
        array = TColgp_Array1OfPnt(1, len(control_points))
        for i, point in enumerate(control_points, 1):
            array.SetValue(i, gp_Pnt(point[0], point[1], 0))
        return Geom_BezierCurve(array)

def main():
    config = {
        'num_shapes': 400,
        'size_range': {
            'min': 700.0,  # Minimum size in mm
            'max': 1000.0,  # Maximum size in mm
            'aspect_ratio_min': 0.5,
            'aspect_ratio_max': 2
        },
        'boundary': {
            'min_points': 4,
            'max_points': 8,
            'min_radius': 350.0,  # in mm
            'max_radius': 450.0,  # in mm
            'angle_variation': 0.15,
            'min_radius_variation': -0.5,
            'max_radius_variation': 0.4,
            'frequency_multiplier': 3,
            'frequency_magnitude': 0.3,
            'inward_curve_probability': 0.2,
            'min_inward_scale': 0.4,
            'max_inward_scale': 0.7,
            'min_self_distance': 30.0  # in mm
        },
        'curvature': {
            'min_radius_factor': 0.2,       # Minimum radius as fraction of segment length
            'length_variation': 0.3,        # Allow 30% variation in control point length
            'max_variation_scale': 0.3,     # Maximum perpendicular variation as fraction of min_radius
        },
        'cutout': {
            'min_size': 60.0,  # in mm
            'max_size': 240.0,  # in mm
            'min_distance_factor': 1.5,
            'max_attempts': 200,
            'circle_probability': 0.5,
            'ellipse_aspect_min': 0.5,
            'ellipse_aspect_max': 1.5,
            'max_cutouts': 1
        },
        'generation': {
            'max_attempts': 50
        },
        'with_cutouts': False,
        'output_dir': r'D:\Projects_Omer\GNN_Project\0_Data\TEST\wo_stiffener\Shapes_OnlyWO\wo_cutout',
        'file_prefix': 'shape_wo_cutout'
    }
    
    generator = OrganicShapeGenerator(config)
    successful = 0
    failed = 0
    shape_index = 0
    
    while successful < config['num_shapes']:
        if generator.create_shape(shape_index):
            successful += 1
            print(f"Generated valid shape {successful}/{config['num_shapes']}")
        else:
            failed += 1
            print(f"Failed to generate shape (attempt {failed})")
        
        shape_index += 1
        
        if failed > config['num_shapes'] * 10:  # 10x buffer for attempts
            print("Maximum attempts reached. Stopping generation.")
            break
    
    print(f"\nGeneration complete:")
    print(f"Successfully generated shapes: {successful}")
    print(f"Failed attempts: {failed}")
    print(f"Success rate: {successful/(successful+failed)*100:.2f}%")

if __name__ == "__main__":
    main()