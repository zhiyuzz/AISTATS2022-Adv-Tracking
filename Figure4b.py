import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from Algorithms.Algorithm_meta import MetaNormBall

# Experiment setting
T = 20000
H = 5
R = 5  # Radius of the domain
d = 1   # Dimension of the domain

# The Lipschitz constants follow from the definition of the loss functions
L = 1
G = H + 1
lam = L * H * (H + 1)

# Hyperparameter
eps_0 = 1

# MetaNormBall is the special case of Algorithm 7 on a norm ball with dimension d and radius R
A = MetaNormBall(lam, eps_0, G, d, R)

# The prediction sequence and the target sequence
predictions = np.empty(T + 1)
x_star = np.empty(T + 1)

for t in range(1, T + 1):
    x_star[t] = signal.square(t * np.pi / 2000)
    predictions[t] = A.get_prediction()
    if predictions[t] == x_star[t]:
        gt = np.zeros(d)
    else:
        gt = (predictions[t] - x_star[t]) / np.linalg.norm(predictions[t] - x_star[t])
    A.update(gt)

plt.rcParams.update({'font.size': 14})
plt.plot(np.arange(1, T), predictions[1:T], '-', label='Predictions $x_t$')
plt.plot(np.arange(1, T), x_star[1:T], '-', label='Target $x^*_t$')

plt.xlabel('t')
plt.legend()
plt.legend(loc='upper left')
