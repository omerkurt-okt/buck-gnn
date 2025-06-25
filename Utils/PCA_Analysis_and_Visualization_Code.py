import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

def generate_L_shape():
    x = np.concatenate([np.linspace(0, 2, 10), np.linspace(0, 1, 5)])
    y = np.concatenate([np.zeros(10), np.linspace(0, 1, 5)])
    
    interior_x = np.random.uniform(0, 2, 30)
    interior_y = np.random.uniform(0, 1, 30)
    mask = ~((interior_x > 1) & (interior_y > 0))
    
    x = np.concatenate([x, interior_x[mask]])
    y = np.concatenate([y, interior_y[mask]])
    
    points = np.column_stack((x, y))
    tri = Delaunay(points)
    return points, tri.simplices

def generate_T_shape():
    x = np.concatenate([np.linspace(-1, 1, 10), np.zeros(5)])
    y = np.concatenate([np.zeros(10), np.linspace(0, 2, 5)])
    
    interior_x = np.random.uniform(-1, 1, 30)
    interior_y = np.random.uniform(0, 2, 30)
    mask = (np.abs(interior_x) < 0.3) | (interior_y < 0.3)
    
    x = np.concatenate([x, interior_x[mask]])
    y = np.concatenate([y, interior_y[mask]])
    
    points = np.column_stack((x, y))
    tri = Delaunay(points)
    return points, tri.simplices

def generate_rectangle_with_hole():
    # Outer rectangle
    x_outer = np.concatenate([np.linspace(-2, 2, 10), np.full(5, 2), 
                            np.linspace(2, -2, 10), np.full(5, -2)])
    y_outer = np.concatenate([np.full(10, -1), np.linspace(-1, 1, 5),
                            np.full(10, 1), np.linspace(1, -1, 5)])
    
    # Inner circle (hole)
    theta = np.linspace(0, 2*np.pi, 20)
    x_inner = 0.5 * np.cos(theta)
    y_inner = 0.5 * np.sin(theta)
    
    # Add random points but avoid the hole
    interior_x = np.random.uniform(-2, 2, 50)
    interior_y = np.random.uniform(-1, 1, 50)
    dist_from_center = np.sqrt(interior_x**2 + interior_y**2)
    mask = dist_from_center > 0.6
    
    x = np.concatenate([x_outer, x_inner, interior_x[mask]])
    y = np.concatenate([y_outer, y_inner, interior_y[mask]])
    
    points = np.column_stack((x, y))
    tri = Delaunay(points)
    return points, tri.simplices

def generate_irregular_shape():
    t = np.linspace(0, 2*np.pi, 30)
    r = 1 + 0.3*np.sin(3*t) + 0.2*np.cos(2*t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    
    interior_points = []
    while len(interior_points) < 40:
        pt = np.random.uniform(-1.5, 1.5, 2)
        r_pt = np.sqrt(pt[0]**2 + pt[1]**2)
        theta_pt = np.arctan2(pt[1], pt[0])
        r_boundary = 1 + 0.3*np.sin(3*theta_pt) + 0.2*np.cos(2*theta_pt)
        if r_pt < r_boundary:
            interior_points.append(pt)
    
    interior_points = np.array(interior_points)
    x = np.concatenate([x, interior_points[:, 0]])
    y = np.concatenate([y, interior_points[:, 1]])
    
    points = np.column_stack((x, y))
    tri = Delaunay(points)
    return points, tri.simplices

def generate_quad_square():
    n = 5  # number of points on each side
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    
    points_x = X.flatten()
    points_y = Y.flatten()
    points = np.column_stack((points_x, points_y))
    
    quads = []
    for i in range(n-1):
        for j in range(n-1):
            p1 = i * n + j
            p2 = p1 + 1
            p3 = (i + 1) * n + j
            p4 = p3 + 1
            quads.append([p1, p2, p4, p3])
    
    return points, np.array(quads)

def generate_quad_rectangle():
    nx, ny = 8, 4  # points in x and y directions
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-0.5, 0.5, ny)
    X, Y = np.meshgrid(x, y)
    
    points_x = X.flatten()
    points_y = Y.flatten()
    points = np.column_stack((points_x, points_y))
    
    quads = []
    for i in range(ny-1):
        for j in range(nx-1):
            p1 = i * nx + j
            p2 = p1 + 1
            p3 = (i + 1) * nx + j
            p4 = p3 + 1
            quads.append([p1, p2, p4, p3])
    
    return points, np.array(quads)

def generate_symmetric_triangle():
    height = 2.0
    base = 2.0
    n_rows = 5
    
    points = []
    quads = []
    point_indices = {}
    current_idx = 0
    
    for i in range(n_rows):
        y = height - (height/n_rows) * i
        points_in_row = 2 * i + 1
        
        if points_in_row == 1:
            points.append([0, y])
            point_indices[(i, 0)] = current_idx
            current_idx += 1
        else:
            row_width = (base/2) * (i/(n_rows-1))
            x_coords = np.linspace(-row_width, row_width, points_in_row)
            
            for j, x in enumerate(x_coords):
                points.append([x, y])
                point_indices[(i, j)] = current_idx
                current_idx += 1
    
    for i in range(n_rows-1):
        points_in_current_row = 2*i + 1
        points_in_next_row = 2*(i+1) + 1
        
        for j in range(points_in_current_row-1):
            p1 = point_indices[(i, j)]
            p2 = point_indices[(i, j+1)]
            
            j1 = int(j * (points_in_next_row-1) / (points_in_current_row-1))
            j2 = int((j+1) * (points_in_next_row-1) / (points_in_current_row-1))
            
            for k in range(j1, j2):
                p3 = point_indices[(i+1, k)]
                p4 = point_indices[(i+1, k+1)]
                quads.append([p1, p2, p4, p3])
    
    return np.array(points), np.array(quads)
def generate_parallelogram():
    # Define parallelogram parameters
    width = 2.0
    height = 1.0
    shear = 0.5
    nx, ny = 6, 4  # points in x and y directions
    
    # Generate grid points
    x = np.linspace(0, width, nx)
    y = np.linspace(0, height, ny)
    X, Y = np.meshgrid(x, y)
    
    # Apply shear transformation
    points_x = X.flatten() + Y.flatten() * shear
    points_y = Y.flatten()
    points = np.column_stack((points_x, points_y))
    
    # Generate quad elements
    quads = []
    for i in range(ny-1):
        for j in range(nx-1):
            p1 = i * nx + j
            p2 = p1 + 1
            p3 = (i + 1) * nx + j
            p4 = p3 + 1
            quads.append([p1, p2, p4, p3])
    
    return points, np.array(quads)

def generate_symmetric_disk():
    # Generate points in concentric circles
    n_circles = 4  # Number of circular layers
    n_points = 16  # Number of points per circle
    points = []
    
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Generate outer ring first (so node 0 will be on the outer ring)
    x_outer = np.cos(theta)
    y_outer = np.sin(theta)
    points.extend(np.column_stack((x_outer, y_outer)))
    
    # Generate inner rings
    for i in range(n_circles - 1, 0, -1):
        radius = i / n_circles
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.extend(np.column_stack((x, y)))
    
    # Add center point last
    points.append([0, 0])
    
    points = np.array(points)
    
    # Generate triangulation
    tri = Delaunay(points)
    return points, tri.simplices

def generate_symmetric_hexagon():
    # Generate regular hexagon vertices first
    n_layers = 4  # Number of layers from center to edge
    points = []
    
    # Hexagon vertices (outer layer)
    hex_points = []
    for i in range(6):
        angle = i * np.pi / 3
        x = np.cos(angle)
        y = np.sin(angle)
        hex_points.append([x, y])
    
    # Add vertices to points list
    points.extend(hex_points)
    
    # Add interior points along hexagon edges
    for layer in range(1, n_layers):
        scale = layer / n_layers
        # Points along edges
        for i in range(6):
            p1 = np.array(hex_points[i]) * scale
            p2 = np.array(hex_points[(i + 1) % 6]) * scale
            # Add points along this edge
            for j in range(1, 3):
                point = p1 + (p2 - p1) * (j / 3)
                points.append(point)
    
    # Add some interior points
    for layer in range(1, n_layers-1):
        scale = layer / n_layers
        for i in range(6):
            angle = (i + 0.5) * np.pi / 3
            x = 0.6 * scale * np.cos(angle)
            y = 0.6 * scale * np.sin(angle)
            points.append([x, y])
    
    # Add center point
    points.append([0, 0])
    
    points = np.array(points)
    
    # Generate triangulation
    tri = Delaunay(points)
    return points, tri.simplices
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
    
    # Create rotation matrix directly from diagonal vector
    rotation = np.array([
        [diagonal[0], -diagonal[1]],
        [diagonal[1], diagonal[0]]
    ]) / np.linalg.norm(diagonal)
    
    # Apply rotation and verify
    transformed_points = centered_points @ rotation
    new_diagonal = transformed_points[p2_idx] - transformed_points[p1_idx]
    print(f"\nAfter rotation:")
    print(f"New diagonal vector: {new_diagonal}")
    print(f"New angle with x-axis: {np.degrees(np.arctan2(new_diagonal[1], new_diagonal[0])):.2f} degrees")
    
    return transformed_points

def transform_pca(points):
    """Transform using PCA for non-symmetric shapes"""
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    transformed = centered_points @ eigenvectors
    moments3 = np.mean(transformed**3, axis=0)
    
    for i in range(2):
        if abs(moments3[i]) > 1e-10 and moments3[i] < 0:
            eigenvectors[:, i] *= -1
            
    return centered_points @ eigenvectors

def transform_to_simulation_coordinates(points):
    """Transform points to simulation coordinates"""
    print(f"\nProcessing shape with {len(points)} points")
    if is_symmetric(points):
        print("Shape detected as symmetric, using diagonal alignment")
        return transform_diagonal_alignment(points)
    else:
        print("Shape detected as non-symmetric, using PCA")
        return transform_pca(points)

def plot_mesh(points, elements, color, alpha, ax, label, marked_node=0, is_quad=False):
    if is_quad:
        for quad in elements:
            quad_points = np.append(quad, quad[0])
            x = points[quad_points, 0]
            y = points[quad_points, 1]
            ax.plot(x, y, color, alpha=alpha)
    else:
        ax.triplot(points[:, 0], points[:, 1], elements, color=color, alpha=alpha)
    
    ax.plot(points[:, 0], points[:, 1], color + '.', markersize=8, label=label)
    
    # Mark and label the specified node
    ax.plot(points[marked_node, 0], points[marked_node, 1], 'r*', markersize=15)
    ax.text(points[marked_node, 0], points[marked_node, 1], f'Node {marked_node}', 
            fontsize=10, ha='right', va='bottom')

def plot_rotated_shape(generator, shape_name, angles, marked_node=0, is_quad=False):
    rows = 2
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(20, 10))
    axs = axs.ravel()
    
    np.random.seed(42)
    original_points, elements = generator()
    
    for i, angle in enumerate(angles):
        theta = np.radians(angle)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])
        
        rotated_points = original_points @ rotation_matrix
        translation = np.random.uniform(-0.5, 0.5, 2)
        rotated_points += translation
        
        transformed_points = transform_to_simulation_coordinates(rotated_points)
        
        ax = axs[i]
        plot_mesh(rotated_points, elements, 'b', 0.3, ax, 'Original', marked_node, is_quad)
        plot_mesh(transformed_points, elements, 'r', 0.3, ax, 'Transformed', marked_node, is_quad)
        
        ax.set_aspect('equal')
        ax.set_title(f'Rotation: {angle}°')
        ax.grid(True)
        ax.legend()
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    
    fig.suptitle(f'{shape_name} - Testing Rotation Invariance', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# Test angles
angles = [0, 30, 41, 72, 96, 239, 299, 112]

# Test all shapes
shapes = [
    (generate_L_shape, "L-Shape", 0, False),
    (generate_T_shape, "T-Shape", 0, False),
    (generate_rectangle_with_hole, "Rectangle with Hole", 0, False),
    (generate_irregular_shape, "Irregular Shape", 0, False),
    (generate_quad_square, "Square Quad Mesh", 0, True),
    (generate_quad_rectangle, "Rectangle Quad Mesh", 0, True),
    (generate_symmetric_triangle, "Y-Symmetric Triangle", 0, True),
    (generate_parallelogram, "Parallelogram", 0, True),
    (generate_symmetric_disk, "Symmetric Disk", 0, False),
    (generate_symmetric_hexagon, "Symmetric Hexagon", 0, False)
]

# Generate plots for each shape
for generator, name, marked_node, is_quad in shapes:
    plot_rotated_shape(generator, name, angles, marked_node, is_quad)