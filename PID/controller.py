# Final Tello UAV controller using PID + yaw compensation
# Based on the structure of your 2D twin-motor control logic
# Author: Youquan Liao
# Collaborator: Shuyan Zhang

import numpy as np

# === Global states ===
prev_error_pos = [0.0, 0.0, 0.0]  # last error [x, y, z] (for Kd)
integral_error_pos = [0.0, 0.0, 0.0]  # Integral error [x, y, z] (for Ki)

prev_error_yaw = 0.0 # last error yaw (for Kd)
integral_error_yaw = 0.0 # Integral error yaw (for Ki)

last_target = [None, None, None, None] # last target position [x, y, z, yaw] (for reset)

def controller(state, target_pos, dt):
    """
    UAV controller: PID (position control) + yaw control + yaw compensation (world to body frame)
    Input:
        state = [x, y, z, roll, pitch, yaw]
        target_pos = [target_x, target_y, target_z, target_yaw]
        dt = timestep (s)
    Output:
        velocity command: (vx, vy, vz, yaw_rate)
    """

    # === Global variables ===
    # PID controller state
    global prev_error_pos, integral_error_pos
    global prev_error_yaw, integral_error_yaw
    global last_target

    # === Extract state ===
    x, y, z, roll, pitch, yaw = state
    target_x, target_y, target_z, target_yaw = target_pos

    # === Reset PID controller ===
    # Reset PID controller if target position changes
    if target_pos != last_target:
        prev_error_pos = [0.0, 0.0, 0.0]
        integral_error_pos = [0.0, 0.0, 0.0]
        prev_error_yaw = 0.0
        integral_error_yaw = 0.0
        last_target = list(target_pos)

    # === PID controller parameters (tuned for Tello) ===
    # PID gains for position control
    # Kp, Ki, Kd for x, y, z
    Kp = 0.25
    Ki = 0.005
    Kd = 0.01

    Kp_yaw = 2.0
    Ki_yaw = 0.02
    Kd_yaw = 0.1

    # === Position error ===
    # Calculate position error in world coordinates
    error_x = target_x - x
    error_y = target_y - y
    error_z = target_z - z

    # === Integral error ===
    # Update integral error
    integral_error_pos[0] += error_x * dt
    integral_error_pos[1] += error_y * dt
    integral_error_pos[2] += error_z * dt

    # === Limit integral error ===
    # Prevent integral windup
    max_integral = 0.5
    for i in range(3):
        integral_error_pos[i] = max(min(integral_error_pos[i], max_integral), -max_integral)

    # === Derivative error ===
    # Calculate derivative error
    derivative_x = (error_x - prev_error_pos[0]) / dt
    derivative_y = (error_y - prev_error_pos[1]) / dt
    derivative_z = (error_z - prev_error_pos[2]) / dt

    # === PID control output (world coordinates) ===
    # Calculate PID control output for x, y, z
    vx = Kp * error_x + Ki * integral_error_pos[0] + Kd * derivative_x
    vy = Kp * error_y + Ki * integral_error_pos[1] + Kd * derivative_y
    vz = Kp * error_z + Ki * integral_error_pos[2] + Kd * derivative_z

    prev_error_pos = [error_x, error_y, error_z]

    # === Limit velocity ===
    # Limit velocity to prevent excessive speed (±0.5 m/s)
    # This is a soft limit, not a hard limit (physical limit: ±1.0 m/s)
    vx = max(min(vx, 0.5), -0.5)
    vy = max(min(vy, 0.5), -0.5)
    vz = max(min(vz, 0.5), -0.5)

    # === Yaw error ===
    # Calculate and normalize yaw error to the range [-pi, pi] in world coordinates
    yaw_error = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

    # === Yaw integral error and derivative error ===
    integral_error_yaw += yaw_error * dt
    derivative_yaw = (yaw_error - prev_error_yaw) / dt

    yaw_rate = (
        Kp_yaw * yaw_error
        + Ki_yaw * integral_error_yaw
        + Kd_yaw * derivative_yaw
    )

    prev_error_yaw = yaw_error

    # === Limit yaw rate ===
    # Limit yaw rate to prevent excessive rotation (±100 deg/s)
    yaw_rate = max(min(yaw_rate, 1.74533), -1.74533)

    # === Convert velocity to body frame ===
    # Convert velocity from world frame to body frame
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    vx_body = cos_yaw * vx + sin_yaw * vy
    vy_body = -sin_yaw * vx + cos_yaw * vy

    # === return velocity command ===
    # Return velocity command in body frame (vx_body, vy_body, vz) and yaw rate
    # vx_body: forward/backward velocity (m/s)
    return (vx_body, vy_body, vz, yaw_rate)
