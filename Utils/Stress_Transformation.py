import math

def stress_transformation(sigma_x, sigma_y, tau_xy, angle_degrees):
    # Convert angle from degrees to radians
    angle = math.radians(angle_degrees)
    
    # Calculate common terms
    cos2 = math.cos(2*angle)
    sin2 = math.sin(2*angle)
    
    # Calculate transformed stresses
    sigma_x_prime = (sigma_x + sigma_y)/2 + ((sigma_x - sigma_y)/2)*cos2 + tau_xy*sin2
    sigma_y_prime = (sigma_x + sigma_y)/2 - ((sigma_x - sigma_y)/2)*cos2 - tau_xy*sin2
    tau_xy_prime = -((sigma_x - sigma_y)/2)*sin2 + tau_xy*cos2
    
    return sigma_x_prime, sigma_y_prime, tau_xy_prime

if __name__ == "__main__":
    sigma_x = 8.488
    sigma_y = -4.23
    tau_xy = -1.17   
    angle = -90.3
    # Calculate transformed stresses
    sx_new, sy_new, txy_new = stress_transformation(sigma_x, sigma_y, tau_xy, angle)
    
    # Print results
    print(f"\nTransformed stresses:")
    print(f"sigma_x_prime = {sx_new:.3f}")
    print(f"sigma_y_prime = {sy_new:.3f}")
    print(f"tau_xy_prime = {txy_new:.3f}")