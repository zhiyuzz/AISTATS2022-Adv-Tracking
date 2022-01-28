import numpy as np
from matplotlib import pyplot as plt
from Algorithms.Algorithm1 import OneDim

# Experiment setting
T = 200

# The loss function is l_t(x) = ||x - x_star||, 1-Lipschitz
x_star = 10
Radius = 15

lam1 = 0.1
lam2 = 0.5
lam3 = 1
gamma = 0
epsilon = 1
Lipschitz = 1

# Create copies of Algorithm 1
A1 = OneDim(lam1, gamma, epsilon, Lipschitz, Radius)
A2 = OneDim(lam2, gamma, epsilon, Lipschitz, Radius)
A3 = OneDim(lam3, gamma, epsilon, Lipschitz, Radius)

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
    if predictions1[t] >= x_star:
        gt = 1
    else:
        gt = -1
    A1.update(gt)

    if predictions2[t] >= x_star:
        gt = 1
    else:
        gt = -1
    A2.update(gt)

    if predictions3[t] >= x_star:
        gt = 1
    else:
        gt = -1
    A3.update(gt)

plt.rcParams.update({'font.size': 14})
plt.plot(np.arange(1, T + 1), predictions1, '-', label='$\lambda=0.1$')
plt.plot(np.arange(1, T + 1), predictions2, '-', label='$\lambda=0.5$')
plt.plot(np.arange(1, T + 1), predictions3, '-', label='$\lambda=1$')

plt.xlabel('t')
plt.ylabel('$x_t$')
plt.legend()
