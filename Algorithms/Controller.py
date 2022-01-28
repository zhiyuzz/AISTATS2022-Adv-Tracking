import numpy as np
from Algorithms.Algorithm_meta_shifted import MetaNormBallShifted


class AdvTracking:
    # Algorithm 5 in the paper, using Algorithm 7 as the base algorithm
    # The implemented version is the special case where l^*_t(x)=||x-x^*_t||
    def __init__(self, A, B, dx, du, kappa, gamma, U, L_star, H, eps0):
        # Store the system dynamics; A, B are functions of t, given at the beginning
        self.A = A
        self.B = B

        self.dx = dx
        self.du = du
        self.H = H

        # Initialize the OCOM algorithm
        L = kappa * L_star
        G = 2 * kappa * L_star / gamma
        lam = L * H * (H + 1)
        self.BaseAlg = MetaNormBallShifted(lam, eps0, G, du, U)

        # Initialize the clock
        self.t = 1

        # Store the past H disturbances
        self.disturbances_past = np.zeros([H, dx])

        # Store the current action
        self.ut = np.zeros(du)

    # Observing the state x_t is equivalent to observing w_{t-1}
    def observe_prev_disturbance(self, w):
        self.disturbances_past = np.roll(self.disturbances_past, -1, axis=0)
        self.disturbances_past[-1] = w

    def get_action(self):
        self.ut = self.BaseAlg.get_prediction()
        return self.ut

    # The received loss functions are ||x_t-x^*_t||
    # For simplicity of implementation, the controller receives the target x^*_t
    def update(self, x_star):
        # Compute the ideal state evaluated at ut, yt(ut)
        # We decompose it into two parts: yt(ut) = from_w + coefficient_u * ut
        from_w = np.zeros(self.dx)
        coefficient_u = np.zeros([self.dx, self.du])
        prod_A = np.identity(self.dx)
        for i in range(1, self.H + 1):  # i is the index t-i from the paper
            if i > 1:
                prod_A = prod_A @ self.A(self.t - i + 1)
            from_w = from_w + prod_A @ self.disturbances_past[-i]
            coefficient_u = coefficient_u + prod_A @ self.B(self.t - i)
        ideal_state = from_w + (coefficient_u @ self.ut)

        direction = (ideal_state - x_star) / np.linalg.norm(ideal_state - x_star)
        gt = direction @ coefficient_u
        self.BaseAlg.update(gt)

        # Update the clock
        self.t += 1
