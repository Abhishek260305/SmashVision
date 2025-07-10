import numpy as np
from numpy.linalg import inv

class KalmanFilter:
    """
    2D Constant Acceleration Kalman Filter for (x, y) tracking.
    """

    def __init__(self, xinit=0, yinit=0, fps=30,
                 std_a=0.001, std_x=0.0045, std_y=0.01, cov=1e5):
        self.dt = 1 / fps

        # State Vector: [x, vx, ax, y, vy, ay]
        self.S = np.array([xinit, 0, 0, yinit, 0, 0], dtype=float)

        # State Transition Matrix (Newtonian Kinematics)
        dt, dt2 = self.dt, self.dt ** 2
        self.F = np.array([
            [1, dt, 0.5 * dt2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])

        # Process Noise Covariance
        q = std_a ** 2
        q11 = 0.25 * dt2 * dt2 * q
        q13 = 0.5 * dt2 * q
        q12 = 0.5 * dt2 * dt * q
        q22 = dt2 * q
        q23 = dt * q
        q33 = q

        self.Q = np.array([
            [q11, q12, q13, 0, 0, 0],
            [q12, q22, q23, 0, 0, 0],
            [q13, q23, q33, 0, 0, 0],
            [0, 0, 0, q11, q12, q13],
            [0, 0, 0, q12, q22, q23],
            [0, 0, 0, q13, q23, q33]
        ])

        # Measurement Noise Covariance
        self.R = np.array([
            [std_x ** 2, 0],
            [0, std_y ** 2]
        ])

        # Observation Matrix (we observe x and y)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

        # Estimate Covariance Matrix
        self.P = np.eye(6) * cov

        # Identity matrix for updates
        self.I = np.eye(6)

        # Initialize histories
        self.S_hist = [self.S.copy()]
        self.P_hist = [self.P.copy()]
        self.K_hist = []

    def predict(self):
        """Prediction Step."""
        self.S_pred = self.F @ self.S
        self.P_pred = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Update Step.
        z: measurement vector [x, y] or [None, None] if no observation.
        """
        if z == [None, None]:
            self.S = self.S_pred
            self.P = self.P_pred
        else:
            z = np.array(z)
            # Kalman Gain
            S_inv = inv(self.H @ self.P_pred @ self.H.T + self.R)
            K = self.P_pred @ self.H.T @ S_inv
            self.K_hist.append(K)

            # Update State
            y = z - self.H @ self.S_pred
            self.S = self.S_pred + K @ y

            # Update Covariance
            self.P = (self.I - K @ self.H) @ self.P_pred

        # Log history
        self.S_hist.append(self.S.copy())
        self.P_hist.append(self.P.copy())

    def step(self, z):
        """Convenience method for one predict-update cycle."""
        self.predict()
        self.update(z)

def cost_fun(a, b):
    """
    Euclidean distance cost function for assignment.
    """
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)

# -------------------------------
# Example main usage
# -------------------------------
if __name__ == "__main__":
    kf = KalmanFilter(xinit=0, yinit=0)

    measurements = [[1, 1], [2, 2], [3, 3], [None, None], [5, 5]]

    for z in measurements:
        kf.step(z)
        print(f"Updated State: x={kf.S[0]:.3f}, y={kf.S[3]:.3f}")
