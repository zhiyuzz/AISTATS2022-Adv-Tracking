import numpy as np
from matplotlib import pyplot as plt
from Algorithms.Algorithm1 import OneDim

# Experiment setting
T = 200

# The loss function is l_t(x) = ||x - x_star||, 1-Lipschitz
x_star_default = 10
x_star_alt = 5

Radius_default = 15
Radius_alt = 50

lam = 1
gamma = 0
epsilon = 1
Lipschitz = 1

# Create copies of Algorithm 1
A1 = OneDim(lam, gamma, epsilon, Lipschitz, Radius_default)
A2 = OneDim(lam, gamma, epsilon, Lipschitz, Radius_default)
A3 = OneDim(lam, gamma, epsilon, Lipschitz, Radius_alt)

# Prediction sequences
predictions1 = np.empty(T)
predictions2 = np.empty(T)
predictions3 = np.empty(T)

for t in range(T):
    # Get predictions
    predictions1[t] = A1.get_prediction()
    predictions2[t] = A2.get_prediction()
    predictions3[t] = A3.get_prediction()

    # Update
    if predictions1[t] >= x_star_alt:
        gt = 1
    else:
        gt = -1
    A1.update(gt)

    if predictions2[t] >= x_star_default:
        gt = 1
    else:
        gt = -1
    A2.update(gt)

    if predictions3[t] >= x_star_default:
        gt = 1
    else:
        gt = -1
    A3.update(gt)

plt.rcParams.update({'font.size': 14})
plt.plot(np.arange(1, T + 1), predictions1, '-', label='$x^*=5$, $R=15$')
plt.plot(np.arange(1, T + 1), predictions3, '-', label='$x^*=10$, $R=50$')
plt.plot(np.arange(1, T + 1), predictions2, '-', label='$x^*=10$, $R=15$')

plt.xlabel('t')
plt.ylabel('$x_t$')
plt.legend()
