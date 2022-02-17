import numpy as np
from matplotlib import pyplot as plt
from Algorithms.Controller import AdvTracking

plt.rcParams['text.usetex'] = True

# Experiment setting
T = 20000

# Hyperparameters
H = 8
eps0 = 0.5


# Dynamics
def dyn_a(time):
    return np.array([[0.55]]) + 0.05 * np.cos(time * np.pi / 10000)


def dyn_b(time):
    return np.array([[0.95]]) + 0.05 * np.cos(time * np.pi / 5000)


# Disturbance
def disturbance(time):
    return np.array([[0]]) + 0.05 * np.sin(time * np.pi / 4000)


# Other problem constants
dx = 1
du = 1
U = 5
kappa = 1
gamma = 0.4
L_star = 1

AdvController = AdvTracking(dyn_a, dyn_b, dx, du, kappa, gamma, U, L_star, H, eps0)

states = np.zeros([T + 1, dx])
x_star = np.empty([T + 1, dx])

# The past disturbance
w_past = disturbance(0)  # w_0

# The past action
past_actual_action = np.zeros(du)   # u_0

# Define a queue for possibly delayed actions (no delay here)
action_Q = np.zeros([1, du])

for t in range(1, T + 1):
    states[t] = dyn_a(t - 1) @ states[t - 1] + dyn_b(t - 1) @ past_actual_action + w_past
    AdvController.observe_prev_disturbance(w_past)

    action_Q[0] = AdvController.get_action()
    past_actual_action = action_Q[-1]
    action_Q = np.roll(action_Q, 1, axis=0)

    x_star[t] = np.sin(t * np.pi / 5000)
    AdvController.update(x_star[t])

    w_past = disturbance(t)

plt.rcParams.update({'font.size': 14})
plt.plot(np.arange(1, T + 1), states[1:T+1], '-', label='States $x_t$')
plt.plot(np.arange(1, T + 1), x_star[1:T+1], '-', label='Target $x^*_t$')

plt.xlabel('t')
plt.legend()
plt.legend(loc='upper left')

plt.savefig("Figures/6c.pdf")
