import numpy as np
    
import math

def calculate_stiffener_contributions(node_id, connected_cbars, node_coords, bdf_model, transformation_matrix):
    """
    Calculate stiffener contributions in 4 main directions:
    0/180, 45/225, 90/270, and 135/315 degrees
    
    Parameters:
    -----------
    node_id : int
        The ID of the current node being processed
    connected_cbars : list
        List of CBAR elements connected to this node
    node_coords : numpy.ndarray
        Coordinates of the current node
    bdf_model : BDF
        The Nastran BDF model
    transformation_matrix : numpy.ndarray
        The transformation matrix for converting to global coordinates
    """
    # Initialize stiffener contributions for each direction
    stiffener_bins = np.zeros(4)  # [0/180, 45/225, 90/270, 135/315]
    ANGLE_TOLERANCE = 1.0  # degree tolerance
    
    for cbar in connected_cbars:
        if cbar.pid == 900:  # Activated element
            # Get the other node
            other_node_id = cbar.nodes[0] if cbar.nodes[1] == node_id else cbar.nodes[1]
            other_coords = np.array(bdf_model.nodes[other_node_id].xyz[:2])
            
            # Calculate direction vector in original coordinates
            direction = other_coords - node_coords
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Transform direction to global coordinates
            transformed_direction = direction @ transformation_matrix
            
            # Calculate angle in degrees (0 to 360)
            angle = np.degrees(np.arctan2(transformed_direction[1], transformed_direction[0])) % 360
            
            # Normalize angle to 0-180 range (since direction doesn't matter)
            if angle > 180:
                angle -= 180
                
            # Define bin centers
            bin_centers = np.array([0, 45, 90, 135])
            
            # Calculate angular distances to each bin center
            distances = np.abs(angle - bin_centers)
            distances = np.minimum(distances, 180 - distances)  # Account for circular nature
            
            # Check if the angle is within tolerance of any bin center
            min_distance = np.min(distances)
            if min_distance <= ANGLE_TOLERANCE:
                # Assign fully to the nearest bin
                nearest_bin = np.argmin(distances)
                stiffener_bins[nearest_bin] += 1.0
            else:
                # Find the two nearest bins
                nearest_indices = np.argsort(distances)[:2]
                
                # Calculate weights based on angular distance
                d1, d2 = distances[nearest_indices]
                total_distance = d1 + d2
                
                # Calculate weights as proportions of the complementary distances
                w1 = d2 / total_distance
                w2 = d1 / total_distance
                
                # Distribute the stiffener contribution to the two nearest bins
                stiffener_bins[nearest_indices[0]] += w1
                stiffener_bins[nearest_indices[1]] += w2

    return stiffener_bins

def transform_to_simulation_coordinates(points):
    """Transform points to simulation coordinates"""
    print(f"\nProcessing shape with {len(points)} points")
    if is_symmetric(points):
        print("Shape detected as symmetric, using diagonal alignment")
        return transform_diagonal_alignment(points)
    else:
        print("Shape detected as non-symmetric, using PCA")
        return transform_pca(points)

def is_symmetric(points, tolerance=1e-6):
    """Check if shape is symmetric by comparing eigenvalues"""
    centered_points = points - np.mean(points, axis=0)
    cov_matrix = np.cov(centered_points.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    ratio = abs(eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + eigenvalues[1])
    print(f"Symmetry check - eigenvalues: {eigenvalues}, ratio: {ratio}")
    return ratio < tolerance

def transform_diagonal_alignment(points):
    """Transform shape by aligning diagonal with x-axis"""
    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Find all pairs of points and their distances
    n_points = len(points)
    distances = []
    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = np.linalg.norm(centered_points[i] - centered_points[j])
            distances.append((dist, i, j))
    
    # Sort by distance to find the longest diagonals
    distances.sort(reverse=True)
    print("\nLongest diagonals:")
    for i in range(min(3, len(distances))):
        dist, idx1, idx2 = distances[i]
        print(f"Distance {dist:.4f} between points {idx1} and {idx2}")
    
    # Use the longest diagonal
    max_dist, p1_idx, p2_idx = distances[0]
    
    # Get diagonal vector
    p1 = centered_points[p1_idx]
    p2 = centered_points[p2_idx]
    diagonal = p2 - p1
    
    print(f"\nSelected diagonal:")
    print(f"P1: {p1}")
    print(f"P2: {p2}")
    print(f"Diagonal vector: {diagonal}")
    
    # Calculate angle between diagonal and x-axis
    angle = np.arctan2(diagonal[1], diagonal[0])
    
    # Create standard rotation matrix (counterclockwise rotation by -angle)
    cos_theta = np.cos(-angle)
    sin_theta = np.sin(-angle)
    rotation = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # Apply rotation and verify
    transformed_points = centered_points @ rotation
    print(f"\nAfter rotation:")
    print(math.degrees(angle))
    
    return transformed_points, centroid, rotation, None

def transform_pca(points):
    """Transform using PCA for non-symmetric shapes"""
    centroid = np.mean(points, axis=0)
    print(centroid)
    centered_points = points - centroid
    
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    transformed = centered_points @ eigenvectors
    moments3 = np.mean(transformed**3, axis=0)
    actual_rotation_angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    print(f"First PCA Transformation angle: {math.degrees(-actual_rotation_angle)} degrees")
    # Track flipping operations
    flip_x = False
    flip_y = False
    
    # Check if we need to flip axes based on third moments
    flip_matrix = np.eye(2)

    for i in range(2):
        if abs(moments3[i]) > 1e-10 and moments3[i] < 0:
            eigenvectors[:, i] *= -1
            if i == 0:
                flip_x = True
            else:
                flip_y = True

    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    print(f"Second PCA Transformation angle: {math.degrees(-angle)} degrees")
    
    # The eigenvectors after moments3 correction form our rotation matrix
    rotation = eigenvectors
    transformed_points = centered_points @ rotation

    transform_info = {
        'rotation_angle': angle,
        'flip_x': flip_x,
        'flip_y': flip_y,
    }
    
    print(f"\nPCA Analysis:")
    print(f"Original rotation angle: {math.degrees(angle)} degrees")
    print(f"Flips: x={flip_x}, y={flip_y}")
    
    return transformed_points, centroid, rotation, transform_info