import numpy as np

# Q = np.diag([1/1.0**2, 1/1.0**2, 1/0.5**2, 1/(np.pi/4)**2])  
# R = np.diag([1/2.0**2, 1/2.0**2, 1/1.0**2, 1/(np.pi/2)**2])  

last_target = [None, None, None, None]

def solve_are(A, B, Q, R, max_iter=100, eps=1e-6):
    """
    Solve the continuous-time algebraic Riccati equation (CARE) using the iterative method.

    Args:
        A (np.ndarray): State matrix.
        B (np.ndarray): Input matrix.
        Q (np.ndarray): State cost matrix.
        R (np.ndarray): Input cost matrix.
        max_iter (int): Maximum number of iterations.
        eps (float): Convergence tolerance.

    Returns:
        np.ndarray: Solution to the CARE.
    """
    P = Q.copy()
    for _ in range(max_iter):
        Pn = A.T @ P @ A - (A.T @ P @ B) @ np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A) + Q
        if np.linalg.norm(Pn - P, ord='fro') < eps:
            break
        P = Pn
    return P

# Implement a controller
def controller(state, target_pos, dt):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    
    global last_target

    if target_pos != last_target:
        last_target = list(target_pos)
        print("Target position updated to:", last_target)
    
    x, y, z, _, _, yaw = state
    target_x, target_y, target_z, target_yaw = target_pos

    e_x = target_x - x
    e_y = target_y - y
    e_z = target_z - z
    e_yaw = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi # normalize yaw error to [-pi, pi]
    error = np.array([e_x, e_y, e_z, e_yaw])
    
    # Define system dynamics matrices
    A = np.eye(4)
    B = -dt * np.eye(4)
    
    # Solve the discrete-time algebraic Riccati equation (DARE)
    deviation_penalty_xy = 0.8
    deviation_penalty_z = 1.0
    deviation_penalty_rpy = 1.0
    control_penalty_xyz = 1.0
    control_penalty_rpy = 0.5

    Q = np.diag([
        deviation_penalty_xy, 
        deviation_penalty_xy, 
        deviation_penalty_z, 
        deviation_penalty_rpy
        ])
    R = np.diag([
        control_penalty_xyz, 
        control_penalty_xyz, 
        control_penalty_xyz, 
        control_penalty_rpy
        ])
    

    P = solve_are(A, B, Q, R)

    # Compute the optimal feedback gain
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    # Compute the control input
    u = -K @ error

    # limit the control input
    u[0] = np.clip(u[0], -0.5, 0.5)
    u[1] = np.clip(u[1], -0.5, 0.5)
    u[2] = np.clip(u[2], -0.5, 0.5)
    u[3] = np.clip(u[3], -np.pi/2, np.pi/2)

    print("Control input:", u)

    output = (u[0], u[1], u[2], u[3])

    return output