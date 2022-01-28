import numpy as np


class OneDim:
    # Algorithm 1 in the paper
    def __init__(self, lam, gamma, eps, G, R_bar):

        # Store the hyperparameters
        self.lam = lam  # lambda
        self.gamma = gamma  # lambda
        self.R_bar = R_bar  # The domain for optimization is the 1d interval [0, R_bar]
        self.C = lam + gamma + G

        # Initialize internal variables from Line 1 of Algorithm 1; assign initial values (t = 1)
        self.t = 1  # time
        self.wealth_past = eps  # Wealth_t
        self.beta_raw = 0  # hat beta_t
        self.beta = 0  # beta_t
        self.prediction_raw = 0  # hat x_t
        self.prediction = 0  # x_t

        # Initialize other interval variables; assign them as 0 (placeholder)
        self.beta_next_raw = 0  # hat beta_{t+1}
        self.beta_next = 0  # beta_{t+1}
        self.wealth = 0  # the current wealth Wealth_t

    def get_prediction(self):
        return self.prediction  # Line 3 - predict

    def update(self, g_t):

        # Line 3 - define the surrogate loss
        if g_t * self.prediction_raw >= g_t * self.prediction:
            loss_surrogate = g_t
        else:
            loss_surrogate = 0

        # Line 4 - compute the next betting fraction
        self.beta_next_raw = (1 - 1 / self.t) * self.beta_raw - loss_surrogate / (2 * self.t * self.C ** 2)
        if self.beta_next_raw < 0:
            self.beta_next = 0
        elif self.beta_next_raw > 1 / (self.C * np.sqrt(2 * self.t)):
            self.beta_next = 1 / (self.C * np.sqrt(2 * self.t))
        else:
            self.beta_next = self.beta_next_raw

        # Line 5 - compute the wealth
        weight_sum = loss_surrogate + self.lam + self.gamma / np.sqrt(self.t)   # add tilde g_t, lambda, gamma/sqrt(t)
        wealth_temp = (1 - weight_sum * self.beta) * self.wealth_past / (1 - self.lam * self.beta_next)
        if self.prediction_raw >= self.beta_next * wealth_temp:
            self.wealth = wealth_temp
        else:
            weight_sum = loss_surrogate - self.lam + self.gamma / np.sqrt(self.t)   # change the weight
            self.wealth = (1 - weight_sum * self.beta) * self.wealth_past / (1 + self.lam * self.beta_next)

        # Update the clock
        self.t += 1
        self.wealth_past = self.wealth
        self.beta_raw = self.beta_next_raw
        self.beta = self.beta_next

        # Line 6
        self.prediction_raw = self.beta * self.wealth_past
        if self.prediction_raw > self.R_bar:
            self.prediction = self.R_bar
        else:
            self.prediction = self.prediction_raw
