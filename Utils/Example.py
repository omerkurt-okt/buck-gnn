import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, PathPatch, Arc
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.special import comb

# Set formal thesis style
def set_publication_style():
    """Set up the plotting style"""
    plt.style.use('seaborn-v0_8-paper')

    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['figure.titlesize'] = 26

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

def bernstein_poly(i, n, t):
    return comb(n, i) * t**i * (1-t)**(n-i)

def bezier_curve(points, num=200):
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t), points[i])
    return curve

def generate_shape():
    fig = plt.figure(figsize=(15, 12))
    
    # Panel A: Initial Point Generation (remains the same)
    ax1 = plt.subplot(221)
    base_radius = 400
    num_points = 8
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    angles += np.random.uniform(-0.15, 0.15, num_points)
    
    radii = base_radius * (1 + np.random.uniform(-0.5, 0.4, num_points))
    radii *= (1 + 0.3 * np.sin(3 * angles))
    inward_mask = np.random.random(num_points) < 0.2
    radii[inward_mask] *= np.random.uniform(0.4, 0.7, np.sum(inward_mask))
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    circle = Circle((0, 0), base_radius, fill=False, color='gray', linestyle='--')
    ax1.add_patch(circle)
    ax1.scatter(x, y, c='black', s=50, zorder=3, label='Boundary Points')
    
    for i in range(num_points):
        ax1.plot([0, x[i]], [0, y[i]], 'r:', alpha=0.5)
        if i % 2 == 0:
            radius = np.sqrt(x[i]**2 + y[i]**2)
            mid_x, mid_y = x[i]/2, y[i]/2
            ax1.annotate(f'{radius:.0f}mm', (mid_x, mid_y), 
                        xytext=(5, 5), textcoords='offset points')
    
    highlight_idx = 2
    ax1.plot([x[highlight_idx], x[(highlight_idx+1)%num_points]], 
             [y[highlight_idx], y[(highlight_idx+1)%num_points]], 
             'g-', linewidth=2, label='Referenced Edge')
    
    ax1.set_title('A) Boundary Point Generation')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()
    
    # Panel B: Single Smooth Bezier Curve Construction
    ax2 = plt.subplot(222)
    
    # Get four consecutive points for the smooth transition
    idx = highlight_idx
    prev_point = points[idx-1]
    p1 = points[idx]
    p2 = points[(idx+1)%num_points]
    next_point = points[(idx+2)%num_points]
    
    # Calculate smooth control points
    base_dir = p2 - p1
    length = np.linalg.norm(base_dir)
    min_radius = length * 0.2
    
    prev_dir = p1 - prev_point
    next_dir = next_point - p2
    
    prev_dir_norm = prev_dir / np.linalg.norm(prev_dir)
    next_dir_norm = next_dir / np.linalg.norm(next_dir)
    base_dir_norm = base_dir / length
    
    # Calculate control points
    base_length = min_radius * (4/3)
    entry_dir = (prev_dir_norm + base_dir_norm)
    exit_dir = (base_dir_norm + next_dir_norm)
    
    entry_dir = entry_dir / np.linalg.norm(entry_dir) * base_length
    exit_dir = exit_dir / np.linalg.norm(exit_dir) * base_length
    
    c1 = p1 + entry_dir
    c2 = p2 - exit_dir
    
    # Create and plot Bezier curve
    curve_points = np.vstack((p1, c1, c2, p2))
    curve = bezier_curve(curve_points)
    
    # Plot previous and next points (faded)
    ax2.scatter([prev_point[0], next_point[0]], 
                [prev_point[1], next_point[1]], 
                c='purple', s=50, alpha=0.7, label='Adjacent Points')
    
    # Plot main segment
    ax2.plot(curve[:, 0], curve[:, 1], 'g-', linewidth=2, label='Bezier Curve')
    ax2.plot([p1[0], c1[0], c2[0], p2[0]], 
             [p1[1], c1[1], c2[1], p2[1]], 
             'r--', alpha=0.7, label='Control Polygon')
    
    # Plot points
    ax2.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                c='black', s=50, label='Boundary Points')
    ax2.scatter([c1[0], c2[0]], [c1[1], c2[1]], 
                c='red', s=50, label='Control Points')
    
    # Add labels
    for point, label in [(p1, 'P1'), (p2, 'P2'), (c1, 'C1'), (c2, 'C2'),
                        (prev_point, 'P_prev'), (next_point, 'P_next')]:
        ax2.annotate(label, point, xytext=(5, 5), textcoords='offset points')
    
    ax2.set_title('B) Smooth Bezier Curve Construction')
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend()
    
    # Panel C: Complete Shape with Linear Edges
    ax3 = plt.subplot(223)
    
    # Plot linear edges
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i+1) % len(points)]
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)
    
    # Add elliptical cutout
    cutout_center = np.array([0, 0])
    ellipse = Ellipse(cutout_center, width=200, height=120, 
                     angle=30, fill=False, color='red', linestyle='--')
    ax3.add_patch(ellipse)
    
    ax3.set_title('C) Complete Shape with Linear Edges')
    ax3.axis('equal')
    ax3.grid(True)
    
    # Panel D: Final Smooth Shape
    ax4 = plt.subplot(224)
    
    # Plot original linear edges
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i+1) % len(points)]
        ax4.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', alpha=0.3)
    
    # Generate and plot smooth curves
    for i in range(len(points)):
        prev_point = points[i-1]
        p1 = points[i]
        p2 = points[(i+1) % len(points)]
        next_point = points[(i+2) % len(points)]
        
        # Calculate smooth control points
        base_dir = p2 - p1
        length = np.linalg.norm(base_dir)
        min_radius = length * 0.2
        
        prev_dir = p1 - prev_point
        next_dir = next_point - p2
        
        prev_dir_norm = prev_dir / np.linalg.norm(prev_dir)
        next_dir_norm = next_dir / np.linalg.norm(next_dir)
        base_dir_norm = base_dir / length
        
        base_length = min_radius * (4/3)
        entry_dir = (prev_dir_norm + base_dir_norm)
        exit_dir = (base_dir_norm + next_dir_norm)
        
        entry_dir = entry_dir / np.linalg.norm(entry_dir) * base_length
        exit_dir = exit_dir / np.linalg.norm(exit_dir) * base_length
        
        c1 = p1 + entry_dir
        c2 = p2 - exit_dir
        
        curve_points = np.vstack((p1, c1, c2, p2))
        curve = bezier_curve(curve_points)
        # Use green color for the highlighted edge, blue for others
        color = 'g' if i == highlight_idx else 'b'
        ax4.plot(curve[:, 0], curve[:, 1], f'{color}-', linewidth=2)
    
    # Add elliptical cutout
    ellipse = Ellipse(cutout_center, width=200, height=120, 
                     angle=30, fill=False, color='red', linestyle='--')
    ax4.add_patch(ellipse)
    
    ax4.set_title('D) Final Shape with Smooth Transitions')
    ax4.axis('equal')
    ax4.grid(True)
    
    # Update legend to include the highlighted curve
    ax4.plot([], [], 'r--', alpha=0.3, label='Original Edges')
    ax4.plot([], [], 'b-', linewidth=2, label='Smooth Shape')
    ax4.plot([], [], 'g-', linewidth=2, label='Referenced Edge')
    ax4.plot([], [], 'r--', label='Elliptical Cutout')
    ax4.legend()

    plt.tight_layout()
    import os
    output_dir = r"D:\Projects_Omer\GNN_Project\ScreenShots\W_Stiffener"
    plt.savefig(os.path.join(output_dir, 'shape_generation_detailed.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'shape_generation_detailed.png'), 
                dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

generate_shape()