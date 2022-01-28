import numpy as np


class OGDBall:
    # Online gradient descent on a norm ball
    # d: dimension of the ball; R: radius; G: Lipschitz constant; scaling: scaling factor of the learning rate
    def __init__(self, d, G, R, scaling):
        self.R = R
        self.G = G
        self.scaling = scaling
        self.prediction = np.zeros(d)
        self.t = 1

    def get_prediction(self):
        return self.prediction

    # Projected gradient step
    def update(self, g_t):
        temp = self.prediction - g_t * self.scaling * self.R / self.G / np.sqrt(self.t)
        if np.linalg.norm(temp) > self.R:
            self.prediction = temp / np.linalg.norm(temp) * self.R
        else:
            self.prediction = temp
        self.t += 1
