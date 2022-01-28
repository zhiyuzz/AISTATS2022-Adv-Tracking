import numpy as np
from Algorithms.Algorithm2_shifted import HigherDimShifted


class SubroutineBallShifted:
    # Subroutine-ball in the paper
    def __init__(self, lam, eps, G, d, R, shift):
        self.lam = lam
        self.G = G
        self.d = d
        self.base = HigherDimShifted(lam, eps, max(lam, G) + G, d, R, shift)
        self.accumulator = np.zeros(d)
        self.base_prediction = self.base.get_prediction()

    def get_prediction(self):
        return self.base_prediction

    def update(self, g_t):
        self.accumulator += g_t
        if np.linalg.norm(self.accumulator) > max(self.lam, self.G):
            self.base.update(self.accumulator)
            self.accumulator = np.zeros(self.d)
            self.base_prediction = self.base.get_prediction()
