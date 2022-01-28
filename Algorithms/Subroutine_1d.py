from Algorithms.Algorithm1 import OneDim


class SubroutineOneD:
    # Subroutine-1d in the paper
    def __init__(self, lam, eps, G):
        self.lam = lam
        self.G = G
        self.base = OneDim(lam, 0, eps, max(self.lam, self.G) + G, 1)
        self.accumulator = 0
        self.base_prediction = self.base.get_prediction()

    def get_prediction(self):
        return self.base_prediction

    def update(self, g_t):
        self.accumulator += g_t
        if abs(self.accumulator) > max(self.lam, self.G):
            self.base.update(self.accumulator)
            self.accumulator = 0
            self.base_prediction = self.base.get_prediction()
