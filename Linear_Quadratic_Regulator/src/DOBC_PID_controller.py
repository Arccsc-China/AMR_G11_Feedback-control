import numpy as np

class PIDWithDisturbanceObserver:
    def __init__(self, Kp, Ki, Kd, Ki_sat, observer_gain):
        # PID coefficients
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.Ki_sat = np.array(Ki_sat)

        # Disturbance observer gain
        self.observer_gain = np.array(observer_gain)

        # Initialize PID state
        self.previous_error = np.zeros(3)
        self.integral = np.zeros(3)

        # Initialize disturbance estimate
        self.estimated_disturbance = np.zeros(3)

    def reset(self):
        """Reset the controller state."""
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.estimated_disturbance = np.zeros(3)

    def update(self, error, measured_output, timestep):
        """
        Update the controller with the current error and measured output.

        Parameters:
        - error: The difference between desired and actual position (3D vector).
        - measured_output: The current output of the system (3D vector).
        - timestep: Time step duration (scalar).

        Returns:
        - control_output: The control signal to be applied (3D vector).
        """

        # Proportional term
        proportional = self.Kp * error

        # Integral term with anti-windup via clamping
        self.integral += error * timestep
        self.integral = np.clip(self.integral, -self.Ki_sat, self.Ki_sat)
        integral = self.Ki * self.integral

        # Derivative term
        derivative = self.Kd * (error - self.previous_error) / timestep

        # PID control signal
        pid_output = proportional + integral + derivative

        # Disturbance estimation
        disturbance_estimation = self.observer_gain * (measured_output - pid_output)
        self.estimated_disturbance += disturbance_estimation * timestep

        # Disturbance compensation
        compensated_control = pid_output - self.estimated_disturbance

        # Update error for next derivative calculation
        self.previous_error = error

        return compensated_control
