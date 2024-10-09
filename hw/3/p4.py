# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#     path: /Users/chan/school/ece595rl/.pixi/envs/default/share/jupyter/kernels/python3
# ---

# ## Problem 4
#
# To model the environment, we can re-use some of our code from HW2. It encodes
# the board, the given policy, the state transition probabilities, and the reward
# function:

# +
import itertools
from enum import Enum, StrEnum, auto

import numpy as np
from IPython.display import Markdown


class BoardSpace(StrEnum):
    NORMAL = "N"
    MOUNTAIN = "M"
    LIGHTNING = "L"
    TREASURE = "T"


class Action(StrEnum):
    UP = "U"
    RIGHT = "R"
    DOWN = "D"
    LEFT = "L"


width = 5

board = np.full([width, width], BoardSpace.NORMAL)

board[2, 1] = BoardSpace.MOUNTAIN
board[3, 1] = BoardSpace.MOUNTAIN
board[1, 3] = BoardSpace.MOUNTAIN
board[2, 3] = BoardSpace.LIGHTNING
board[4, 4] = BoardSpace.TREASURE

policy = np.flipud(
    np.array(
        [
            list("RRRRU"),
            list("LULLU"),
            list("UURRR"),
            list("UDDDU"),
            list("URRUU"),
        ]
    )
).T

gamma = 0.95


def i_1d(x, y):
    return np.ravel_multi_index([y, x], dims=[width, width])


def is_blocked(x, y):
    return (
        x < 0
        or y < 0
        or x >= width
        or y >= width
        or board[x, y] == BoardSpace.MOUNTAIN
    )


def get_trans_prob(x: int, y: int, a: Action):
    if x < 0 or x >= width or y < 0 or y >= width:
        raise RuntimeError("Invalid coordinates")

    if a not in Action:
        raise RuntimeError("Invalid action type")

    p_vec = np.zeros(width**2)
    i_state_1d = i_1d(x, y)

    if board[x, y] != BoardSpace.NORMAL:
        p_vec[i_state_1d] = 1
        return p_vec

    direction_coordinates = {
        Action.LEFT: [x - 1, y],
        Action.RIGHT: [x + 1, y],
        Action.UP: [x, y + 1],
        Action.DOWN: [x, y - 1],
    }

    for direction, (x_next, y_next) in direction_coordinates.items():
        prob = 0.85 if direction == a else 0.05
        if is_blocked(x_next, y_next):
            p_vec[i_state_1d] += prob
        else:
            p_vec[i_1d(x_next, y_next)] += prob

    if p_vec.sum() != 1:
        raise RuntimeError("Probability vector did not add to 1")

    return p_vec


# (0, 0), (1, 0), (2, 0) ... (3, 4), (4, 4)
each_state = [(x, y) for y in range(5) for x in range(5)]


def p_matrix(policy):
    p = np.zeros([width**2, width**2])

    for x, y in each_state:
        i = i_1d(x, y)
        p[i] = get_trans_prob(x, y, policy[x, y])

    return p


reward_fn = np.zeros(width * width, dtype=int)
reward_fn[i_1d(*np.argwhere(board == BoardSpace.LIGHTNING).squeeze())] = -1
reward_fn[i_1d(*np.argwhere(board == BoardSpace.TREASURE).squeeze())] = 1


def vecfmt(v):
    return Markdown(", ".join([f"{e:g}" for e in v]))


# -

# ### 4.1
#
# To ensure that each state is visited often, we need to generate trajectories
# using a wide range of initial states. We could use a uniform random distribution
# to select our initial state for the trajectories, but in order to guarantee that
# all states get visited at least some given number of times, we will simply loop
# over each state (that is not a mountain) and generate the same number of
# trajectories beginning at that state. This ensures that each state is visited at
# least that many times.
#
# We attempt to determine our termination condition using the criterion derived in
# [2.4]. We showed there that
#
# $$
# \mathbb{P}\left(\left| \hat{V}^{\pi}(s) - V^{\pi}(s) \right| \ge \varepsilon'\right) \le 2 \exp \left( -\frac{ (\varepsilon')^{2}N^{s}}{2 \left( R_{max} \frac{1 - \gamma^{|\tau_i^s|+1}}{1-\gamma} \right)^{2}} \right), \quad \forall \varepsilon' \ge 0
# $$
#
# We don't know how long the trajectories will be since they are random, so we can
# assume they are infinitely long, which will give us an even more conservative
# estimate:
#
# $$
# \mathbb{P}\left(\left| \hat{V}^{\pi}(s) - V^{\pi}(s) \right| \ge \varepsilon'\right) \le 2 \exp \left( -\frac{ (\varepsilon')^{2}N^{s}}{2 \left( R_{max} \frac{1}{1-\gamma} \right)^{2}} \right), \quad \forall \varepsilon' \ge 0
# $$
#
# Now, we need to supply some values for the desired deviation between the sample
# mean and its expected value, as well as the probability with which we want to be
# within that bound. We can choose a probability of $0.95$ and a $\varepsilon$ of
# $0.01$, and solve for $N^s$, the number of sample trajectories required to
# achieve this goal. The probability is stated such that it is the probability
# that the absolute difference is *not* within $\varepsilon$, so we need to use
# $1-0.95$ on the LHS of the inequality:
#
# $$
# \begin{aligned}
# 1-0.95 &\le 2 \exp \left( - \frac{0.01^2 N^s}{2 \left( \frac{1}{1-0.95} \right)^2} \right) \\
# \frac{0.05}{2} &\le \exp \left( - \frac{ 0.0001 }{800} N^s \right) \\
# N^s &\ge - 8 \cdot 10^6 \ln \left(\frac{0.05}{2}\right) \\
# N^s &\ge `{python} Markdown(f"{(np.log(0.05 / 2) * 2 * (1 / 0.05)**2 / (-0.01**2)):.5e}")`
# \end{aligned}
# $$
#
# Unfortunately, it appears that this upper bound is *extremely* conservative in
# this case, and we don't have the computational power to perform this many
# iterations. Luckily, through experimentation, it is shown that the value
# function converges long before then, and we choose $5000$ iterations on each
# initial state since the max difference between iterations tails to nearly zero
# by this point, as shown in the plots later.

# +
terminal_spaces = [BoardSpace.LIGHTNING, BoardSpace.TREASURE]

p = p_matrix(policy)

rng = np.random.default_rng(seed=42109581092395879)


def sample_trajectory(initial_state, policy):
    states = []
    actions = []
    rewards = []

    state = initial_state

    while True:
        x, y = state
        action = policy[x, y]
        reward = reward_fn[i_1d(x, y)]

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        if board[*state] in terminal_spaces:
            break

        state_1d = rng.choice(
            np.arange(width**2), p=p[i_1d(x=state[0], y=state[1])]
        )

        y, x = np.unravel_index(state_1d, [width, width])

        state = np.array([x, y])

    return (np.array(vals) for vals in [states, actions, rewards])


# FIXME
n_samples = 500
# n_samples = 5000

vhat_mc = np.zeros([n_samples, width**2])

# Initialize vhat(s), \mathcal{G}(s), N(s)
vhat = np.zeros([width, width], dtype=float)
returns = {state: [] for state in each_state}
num_visits = np.zeros([width, width], dtype=int)

i = 0

reachable_states = [
    state for state in each_state if board[*state] != BoardSpace.MOUNTAIN
]

# for i in tqdm(range(n_samples)):
for i in range(n_samples):
    for initial_state in reachable_states:
        # Generate a sample trajectory using the policy
        states, actions, rewards = sample_trajectory(initial_state, policy)

        # If the pair in question isn't in the trajectory at all, then don't
        # attempt to update vhat
        for state in np.unique(states, axis=0):

            # Obtain the time at which the state is first visited
            t_s = np.argwhere(np.all(states == state, axis=1)).flatten()[0]

            # Get a vector of gamma to the (t - t_s) power for the timesteps from
            # t_s onward in order to calculate the return G
            gammas = np.pow(gamma, np.arange(len(states) - t_s))

            # Calculate G
            ret = float(np.sum(gammas * rewards[t_s:]))

            state_tuple = (int(state[0]), int(state[1]))

            returns[state_tuple].append(ret)

            num_visits[*state] += 1

            # Incremental calculation of the sample mean
            vhat[*state] = float(
                vhat[*state] + (1 / num_visits[*state]) * (ret - vhat[*state])
            )

    vhat_mc[i] = vhat.flatten()

print(f"Number of visits at each state:")
print(np.flipud(num_visits.T))
print(f"\n\nValue function at each state:")
print(np.flipud(vhat.T))
# -

# The value function for each state on the board is shown above, as well as the
# number of visits at each state.
#
# ### 4.2
#
# We implement the one-step TD algorithm below. Similarly to before, we force the
# algorithm to perform the same number of iterations using each reachable state as
# the initial state, guaranteeing that each state is visited at least that number
# of times. Also similarly to before, we use $5000$ iterations since the plots
# below show convergence by this point in time.

# +
# Initialize vhat
vhat = np.zeros([width, width], dtype=float)

# FIXME
n_iterations = 500
# n_iterations = 5000

learning_rate = 0.1

p = p_matrix(policy)

vhat_td = np.zeros([n_iterations, width**2])

# for i in tqdm(range(n_iterations)):
for i in range(n_iterations):
    # Sample an initial state randomly
    for state_i in reachable_states:
        state = np.array(state_i)
        x, y = state
        while True:
            # Obtain the immediate reward
            # reward = reward_fn[i_1d(x, y)]

            if board[*state] in terminal_spaces:
                # Take one more action and receive the reward at the terminal
                # state. Regardless of action we will obtain the same reward for
                # a terminal state.
                reward = reward_fn[i_1d(x, y)]
                v_state = vhat[x, y]

                # There is no new state after the terminal state, so there
                # should be no new value function
                v_state_new = 0

                vhat[x, y] = v_state + learning_rate * (
                    reward + gamma * v_state_new - v_state
                )

                break

            # Obtain the next state according to the transition function under the
            # policy (the correct action was already chosen when the p matrix was
            # calculated, so there is no need to explicitly get it here)
            state_1d_new = rng.choice(np.arange(width**2), p=p[i_1d(x, y)])
            y_new, x_new = np.unravel_index(state_1d_new, [width, width])
            state_new = np.array([x_new, y_new])

            # Collect the reward for the action taken at the current state
            reward = reward_fn[i_1d(x, y)]

            v_state = vhat[x, y]
            v_state_new = vhat[x_new, y_new]

            vhat[x, y] = v_state + learning_rate * (
                reward + gamma * v_state_new - v_state
            )

            state = state_new.copy()
            x, y = state

    vhat_td[i] = vhat.flatten()

print(np.flipud(vhat.T))
# -

# ### 4.3

# +
lightning_coord = np.argwhere(board == BoardSpace.LIGHTNING).squeeze()
treasure_coord = np.argwhere(board == BoardSpace.TREASURE).squeeze()

r = np.zeros(width * width + 1, dtype=int)
r[i_1d(*lightning_coord)] = -1
r[i_1d(*treasure_coord)] = 1

p = np.vstack([p, np.zeros(25)])
p = np.hstack([p, np.zeros(26).reshape(-1, 1)])
p[25, :] = 0
p[:, 25] = 0
p[25, 25] = 1
p[i_1d(*lightning_coord), :] = 0
p[i_1d(*treasure_coord), :] = 0
p[i_1d(*lightning_coord), 25] = 1
p[i_1d(*treasure_coord), 25] = 1

v = (np.linalg.inv(np.identity(width**2 + 1) - gamma * p) @ r)[:-1]


print("Analytical value function solution:")
with np.printoptions(precision=4):
    print(np.flipud(v.reshape([5, 5]).T))
# -

# ### 4.4

# +
import matplotlib.pyplot as plt

mc_err = np.linalg.norm(vhat_mc - v, axis=1)
td_err = np.linalg.norm(vhat_td - v, axis=1)

fig, ax = plt.subplots()
ax.plot(mc_err)
ax.plot(td_err)
fig.savefig("test.png")
fig.show()
