from Algorithms.Algorithm1 import *
from Algorithms.AlgorithmOGD import *


class HigherDimShifted:
    # d: dimension of the domain
    def __init__(self, lam, eps, G, d, R, shift):
        self.shift = shift
        self.A_r = OneDim(lam, lam, eps, G, R + np.linalg.norm(shift))
        self.A_B = OGDBall(d, G, 1, 0.1)
        self.yt = 0
        self.zt = np.empty(d)

    def get_prediction(self):
        self.yt = self.A_r.get_prediction()
        self.zt = self.A_B.get_prediction()
        return self.shift + self.yt * self.zt

    def update(self, g_t):
        self.A_r.update(g_t @ self.zt)
        self.A_B.update(g_t)
