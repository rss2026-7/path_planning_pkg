
import numpy as np
class MotionModel:

    def __init__(self, node):
        self.node = node

        # Motion noise parameters
        self.sigma_x = 0.2
        self.sigma_y = 0.05
        self.sigma_theta = 0.05

        # Deterministic mode: when True, no noise is added (for unit tests)
        node.declare_parameter('deterministic', False)
        self.deterministic = node.get_parameter('deterministic').get_parameter_value().bool_value

        ####################################

    def evaluate(self, particles, odometry):
        dx, dy, dtheta = odometry
        n = particles.shape[0]

        # Current heading of each particle
        theta = particles[:, 2].copy()

        if self.deterministic:
            # No noise — used for unit tests and deterministic simulation
            particles[:, 0] += np.cos(theta) * dx - np.sin(theta) * dy
            particles[:, 1] += np.sin(theta) * dx + np.cos(theta) * dy
            particles[:, 2] = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi
        else:
            # Add Gaussian noise to the odometry for each particle
            noisy_dx = dx + np.random.normal(0.0, self.sigma_x, n)
            noisy_dy = dy + np.random.normal(0.0, self.sigma_y, n)
            noisy_dtheta = dtheta + np.random.normal(0.0, self.sigma_theta, n)

            # Convert body-frame motion into world-frame motion
            particles[:, 0] += np.cos(theta) * noisy_dx - np.sin(theta) * noisy_dy
            particles[:, 1] += np.sin(theta) * noisy_dx + np.cos(theta) * noisy_dy
            particles[:, 2] = (theta + noisy_dtheta + np.pi) % (2 * np.pi) - np.pi

        return particles

        ####################################
