import numpy as np

class KalmanFilter:
    def __init__(self, init_state_vector: np.ndarray=None, init_measurement_vector: np.ndarray=None):
        self.state_vector = init_state_vector
        self.measurement_vector = init_measurement_vector
        self.n = self.state_vector.shape[0]
        self.m = self.measurement_vector.shape[0]
        assert self.n >= self.m, "dim of state vector must be greater or equal\
            dim of measurement vector. Otherwise, expand dim of state vector"
        self.F = np.eye(self.n, dtype=float)
        self.P = np.eye(self.n, dtype=float) * 0.1
        self.H = np.zeros((self.n, self.m), dtype=float)
        for i in range(self.m):
            self.H[i, i] = 1.0
        self.Q = np.eye(self.n, dtype=float) *0.1
        self.R = np.eye(self.m, dtype=float) *0.1
        self.identity_matrix = np.eye(self.n, dtype=float)

        ## This part is a bit tricky

    def predict(self):
        self.x_hat = np.dot(self.F, self.state_vector)
        tmp_result = np.einsum('ij,jk,kl->il', self.F, self.P, self.F.T)
        self.predict_P = tmp_result + self.Q

    def update(self, measurement_vector):
        ## Calculate Kalman gain
        total_uncertainty = np.einsum("ij,jk,kl->il", self.H, self.predict_P, self.H.T) + self.R
        self.K_gain = np.dot(np.einsum('ij,jk->ik', total_uncertainty, self.H.T), np.linalg.inv(total_uncertainty))
        
        # Update state estimate
        residual_measurement = measurement_vector - np.dot(self.H, self.x_hat)
        self.state_vector = self.x_hat + np.dot(self.K_gain, residual_measurement)
        self.P = (self.identity_matrix - np.dot(self.K_gain, self.H)) @ self.predict_P
        new_state = self.state_vector.copy()
        return new_state
