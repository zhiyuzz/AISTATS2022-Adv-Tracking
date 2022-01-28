import numpy as np
from Algorithms.Subroutine_ball import SubroutineBall
from Algorithms.Subroutine_1d import SubroutineOneD


class MetaNormBall:
    # This is the special case of Algorithm 7 with the domain V being a Euclidean norm ball
    # In this case we do not need a projection subroutine
    # lam: the constant defined in Line 1 of Algorithm 7
    # eps_0: the hyperparameter of Algorithm 7
    # G: Lipschitz constant
    # d, R: dimension and radius of the domain

    def __init__(self, lam, eps_0, G, d, R):
        self.lam = lam
        self.eps_0 = eps_0
        self.G = G
        self.d = d
        self.R = R

        self.t = 1  # time
        self.K_plus_one = 1  # K_t defined in Line 4, plus one
        self.ALGs_ball = {}  # The collection of Subroutine-ball
        self.ALGs_1d = {}   # The collection of Subroutine-1d
        self.prediction_temp_tilde = {}   # The temporary predictions tilde x^k_t
        self.prediction_temp_projected = {}  # The temporary predictions x^k_t
        self.gradient_temp = {}  # The temporary gradients sent to the subroutines g^k_t

    def get_prediction(self):
        # The K_t defined Line 4, plus one
        self.K_plus_one = np.ceil(np.log2(self.t + 1)).astype(int)
        self.prediction_temp_tilde = np.zeros([self.K_plus_one + 1, self.d])
        self.prediction_temp_projected = np.zeros([self.K_plus_one + 1, self.d])

        # Line 3
        for k in range(self.K_plus_one):
            if self.t % (2 ** k) == 0:
                self.ALGs_ball[k] = SubroutineBall(self.lam, self.eps_0 * 2 ** k, self.G, self.d, self.R)
                self.ALGs_1d[k] = SubroutineOneD(self.lam * self.R, self.eps_0 * 2 ** k, self.G * self.R)

        # Line 5 to 8
        for k in range(self.K_plus_one - 1, -1, -1):
            w_t = self.ALGs_ball[k].get_prediction()
            z_t = self.ALGs_1d[k].get_prediction()
            temp = (1 - z_t) * self.prediction_temp_projected[k + 1] + w_t
            self.prediction_temp_tilde[k] = temp
            if np.linalg.norm(temp) > self.R:
                self.prediction_temp_projected[k] = temp * self.R / np.linalg.norm(temp)
            else:
                self.prediction_temp_projected[k] = temp

        return self.prediction_temp_projected[0]    # the prediction is x^0_t

    def update(self, g_t):
        self.gradient_temp = np.empty([self.K_plus_one + 1, self.d])
        self.gradient_temp[-1] = g_t
        for k in range(self.K_plus_one):
            # Line 10, 11, 14
            gradient_temp = self.gradient_temp[k - 1]
            if gradient_temp @ self.prediction_temp_tilde[k] >= gradient_temp @ self.prediction_temp_projected[k]:
                self.gradient_temp[k] = self.gradient_temp[k - 1]
            else:
                direction = self.prediction_temp_tilde[k] / np.linalg.norm(self.prediction_temp_tilde[k])
                self.gradient_temp[k] = self.gradient_temp[k - 1] - (self.gradient_temp[k - 1] @ direction) * direction

            # Line 13
            self.ALGs_ball[k].update(self.gradient_temp[k])
            self.ALGs_1d[k].update(- self.gradient_temp[k] @ self.prediction_temp_projected[k + 1])

        # Update the clock
        self.t += 1
