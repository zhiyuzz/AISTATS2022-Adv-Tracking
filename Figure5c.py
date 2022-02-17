import numpy as np
from matplotlib import pyplot as plt
from Algorithms.Algorithm_meta_shifted import MetaNormBallShifted

plt.rcParams['text.usetex'] = True

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
# MetaNormBallShifted is the shifted version of MetaNormBall
A = MetaNormBallShifted(lam, eps_0, G, d, R)

# The prediction sequence and the target sequence
predictions = np.empty(T + 1)
x_star = np.empty(T + 1)

for t in range(1, T + 1):
    x_star[t] = np.sin(t * np.pi / 2000)
    predictions[t] = A.get_prediction()
    if predictions[t] == x_star[t]:
        gt = np.zeros(d)
    else:
        gt = (predictions[t] - x_star[t]) / np.linalg.norm(predictions[t] - x_star[t])
    A.update(gt)

plt.rcParams.update({'font.size': 14})
plt.plot(np.arange(1, T + 1), predictions[1:T+1], '-', label='Predictions $x_t$')
plt.plot(np.arange(1, T + 1), x_star[1:T+1], '-', label='Target $x^*_t$')

plt.xlabel('t')
plt.legend(loc='upper left')

plt.savefig("Figures/5c.pdf")
