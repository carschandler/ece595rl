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

# ---
# title: "ECE59500RL HW2"
# author: "Robert (Cars) Chandler — chandl71@purdue.edu"
# format:
#   pdf
# jupyter: python3
# ---
#
# ## Problem 1
#
# ## Problem 2
#
# ## Problem 3
#
# ## Problem 4
#
# First, we need to assign labels to the states, which are the different spaces on
# the board. We will use a zero-indexed $x$-$y$ coordinate system to refer to the
# different states $s_{x,y} \in \mathcal{S}$, with the origin at the bottom left
# square, $s_{0,0}$. Moving horizontally will increase the $x$-component and
# vertically the $y$-component, so that our state space is 
#
# $$
# \mathcal{S} =\{s_{ij} : i, j \in \mathbb{N}_{0}, \quad i,j \le 5 \}
# $$
#
# ::: {.callout-note}
# ### Note
#
# Although we are using two "dimensions" to identify each state, we still treat
# our state-space as a one-dimensional vector, so that we have one row for each $s
# \in \mathcal{S}$ in $\vec{v}$, $P^{\pi}$, and so forth. The order of the states
# for this vector will always be in row-major order using the same $x$-$y$
# coordinate system described above:
#
# $$
# (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), ..., (3, 4), (4, 4)
# $$
# :::
#
# ### 4.1.a
#
# The policy can be evaluated analytically using the following equation:
#
# $$
# \vec{v}^{\pi} = (I - \gamma P^{\pi})^{-1} \vec{R}^{\pi}
# $$ {#eq-analytical-soln}
#
# We have $\gamma = 0.95$ from the problem statement. $P^{\pi}$ and
# $\vec{R}^{\pi}$ each need to be evaluated by going over each state in
# $\mathcal{S}$ and using the information given to us to evaluate them. Beginning
# with $P^{\pi}$:
#
# $$
# P^{\pi}_{ij} = P(s_{j} | s_{i}, a) = P(s_{j} | s_{i}, \pi(s_{i}))
# $$
#
# We can use Python to encode the logic described in the problem statement to
# programatically calculate $P^{\pi}$ for each state transition:

# +
from enum import Enum, StrEnum, auto
import numpy as np
import itertools
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

policy = np.array(
    [
        list("URRUU"),
        list("UDDDU"),
        list("UURRR"),
        list("LULLU"),
        list("RRRRU"),
    ]
).T


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

p = np.zeros([width**2, width**2])

for x, y in each_state:
    i = i_1d(x, y)
    p[i] = get_trans_prob(x, y, policy[x, y])

state_text = []
for i in range(width**2):
    y1, x1 = np.unravel_index(i, [width, width])
    a = policy[x1, y1]
    for j in np.argwhere(p[i]).flatten():
        y2, x2 = np.unravel_index(j, [width, width])
        state_text.append(
            rf"P^{{\pi}}(s_{{ {x2}, {y2} }} | s_{{ {x1}, {y1} }}, \pi(s_{{"
            rf" {x1},"
            rf" {y1} }}) = \text{{{a}}}) &= {p[i, j]:g} \\"
        )
    state_text.append(r"\\")

state_text = "\n".join(state_text)


def vecfmt(v):
    return Markdown(", ".join([f"{e:g}" for e in v]))


# -

# The resulting state transition probabilities are listed as follows:
#
# \begingroup
# \allowdisplaybreaks
# \begin{align*}
# `{python} Markdown(state_text)`
# \end{align*}
# \endgroup
#
# All other possible state transitions have probability $0$.
#
# Moving onto $\vec{R}^{\pi}$:
#
# $$
# \vec{R}^{\pi} = \begin{bmatrix}
# R(s_{1,1}, \pi(s_{1,1}) \\
# R(s_{1,2}, \pi(s_{1,2}) \\
# \cdots \\
# R(s_{4,4}, \pi(s_{4,4}))
# \end{bmatrix}_{|\mathcal{S}| \times 1}
# $$
#
# So given the reward function described, we just have a simple vector with two
# nonzero elements:

r = np.zeros(width * width, dtype=int)
r[i_1d(*np.argwhere(board == BoardSpace.LIGHTNING).squeeze())] = -1
r[i_1d(*np.argwhere(board == BoardSpace.TREASURE).squeeze())] = 1

# $$
# \vec{R}^{\pi} = \left[ `{python} vecfmt(r)` \right]^T
# $$
#
# So, now we can substitute in our values into @{eq-analytical-soln} and solve:

# +
gamma = 0.95
v = np.linalg.inv(np.identity(width**2) - gamma * p) @ r
value_text = []
for i, val in enumerate(v):
    y, x = np.unravel_index(i, [width, width])
    value_text.append(rf"v^{{\pi}}(s_{{ {x},{y} }}) &= {val:g} \\")

value_text = "\n".join(value_text)
# -

# \begingroup
# \allowdisplaybreaks
# \begin{align*}
# `{python} Markdown(value_text)`
# \end{align*}
# \endgroup
#
# ### 4.1.b
#
# Let $\vec{v}_0 = \mathbf{0}$. Then for each step in the iteration,
#
# $$
# \vec{v}_{t+1} = \vec{R}^{\pi} + \gamma P^{\pi} \vec{v}_{t}
# $$
#
# To determine $T$, the number of iterations we need to make in order to obtain
# $\Vert v_T - v^\pi \Vert_{\infty} \le 0.01$, we can use the following theorem:
#
# $$
# T \ge \frac{\log \left( \frac{\Vert \vec{v}_{0} - \vec{v}^{\pi}\Vert_{\infty} }{\varepsilon} \right)}{\log \frac{1}{\gamma}}
# $$ {#eq-end-criterion}
#
# *However*, we cannot use the analytical solution of $v^{\pi}$, so we need to
# form some kind of bound on it instead and use that. We know that
#
# $$
# v^{\pi}(s) = \mathbb{E} \left[
# \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) | s_0 = s \right], \quad \forall s \in
# \mathcal{S}
# $$
#
# by definition, and we know that our reward function is bounded on the interval
# $[-1, 1]$. Therefore, we can say that $|R(s, a)| \le 1$ so that
#
# $$
# \Vert v^{\pi} \Vert_{\infty} \le 1 \cdot \sum_{t=0}^{\infty} \gamma^t = \frac{1}{1 - \gamma}
# $$
#
# And since $v_0 = \mathbf{0}$, we can say that 
#
# $$
# \Vert \vec{v}_{0} -
# \vec{v}^{\pi}\Vert_{\infty} \le \Vert v^{\pi} \Vert_{\infty} \le \frac{1}{1 - \gamma}
# $$
#
# So, finally:
#
# $$
# \frac{\log \left( \frac{1}{\varepsilon(1 - \gamma)} \right)}{\log \frac{1}{\gamma}} 
# $$
#
# will give us a conservative estimate of $T$ which is more than (or equal to) the
# number of iterations actually required to obtain our desired error.
#
# We can evaluate this result using Python, and then perform $T$ iterations by
# implementing the iteration algorithm above.

# +
from numpy.linalg import norm

epsilon = 0.01
v0 = np.zeros(width**2)
n_iterations = int(
    np.ceil(np.log(1 / (epsilon * (1 - gamma))) / np.log(1 / gamma))
)

v_t = v0
v_history = [v0]
for t in range(n_iterations):
    v_t = r + gamma * p @ v_t
    v_history.append(v_t)

v_history = np.array(v_history)

value_text_iter = []
for i, val in enumerate(v):
    y, x = np.unravel_index(i, [width, width])
    value_text_iter.append(
        rf"v_{{T = {n_iterations}}} (s_{{ {x},{y} }}) &= {val:g} \\"
    )

value_text_iter = "\n".join(value_text_iter)
# -

# We perform $T = `{python} n_iterations`$ iterations, and our final $v_T$ is:
#
# \begingroup
# \allowdisplaybreaks
# \begin{align*}
# `{python} Markdown(value_text_iter)`
# \end{align*}
# \endgroup
#
# We can verify that our desired conditon holds:

max_error = norm(v_t - v, ord=np.inf)

# $$
# \Vert v_T - v^\pi \Vert_{\infty} = `{python} Markdown(f"{max_error:0.6f}")` \le 0.01
# $$
#
# ### 4.1.c
#
# We kept track of the full history of $v_t$, so we can calculate the error for
# each timestep and plot it:

# +
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "png"
pio.kaleido.scope.default_scale = 2

error_history = norm(v_history - v, ord=np.inf, axis=1)
go.Figure(
    data=[go.Scatter(y=error_history, mode="lines")],
    layout=dict(
        xaxis_title="$t$", yaxis_title=r"$\Vert v_t - v^\pi \Vert_{\infty}$"
    ),
)
# -

# ## 4.2
#
# To perform value iteration, we initialize $\vec{v}_0 = \mathbf{0}$ and then for
# each iteration which increases $t$ by one, we perform the operation:
#
# $$
# v_{t+1}(s) = \max_{a \in \mathcal{A}} \left[ R(s,a) + \gamma \mathbb{E}_{s' \sim
# P(\cdot | s,a)} [v_t(s')] \right], \quad \forall s \in \mathcal{S}
# $$
#
# However, since our 
#
# We can use the following theorem to determine when we have completed a
# sufficient number of iterations:
#
# $$
# T \ge \frac{\log \left( \frac{\Vert \vec{v}_{0} - \vec{v}^{*}\Vert_{\infty} }{\varepsilon} \right)}{\log \frac{1}{\gamma}}
# $$
#
# This is the same as @{eq-end-criterion} but with $v^*$ instead of $v^{\pi}$. We
# can use the same logic as before to bound $T$ since the only difference is that
# $v^*$ now considers all possible policies, but all the value functions for all
# policies can still be bound by the reward function as before, such that:
#
# $$
# \Vert v^{*} \Vert_{\infty} \le 1 \cdot \sum_{t=0}^{\infty} \gamma^t = \frac{1}{1 - \gamma}
# $$
#
# Therefore, we can use the same $T$ as before.

# +
v0 = np.zeros(width**2)

v_t = v0

epsilon = 0.01

n_iterations = int(
    np.ceil(np.log(1 / (epsilon * (1 - gamma))) / np.log(1 / gamma))
)

for t in range(n_iterations):
    v_t = np.array(
        [
            np.max(
                [
                    r[i_1d(x, y)]
                    + gamma * np.sum(get_trans_prob(x, y, a) * v_t)
                    for a in Action
                ]
            )
            for x, y in each_state
        ]
    )

print(v_t)
# -

# Now, with $\vec{v}_T$ determined, we need to find the policy corresponding to
# this value function, which is:
#
# $$
# \pi_T(s) = \arg \max_{a \in \mathcal{A}} [R(s,a) + \gamma \mathbb{E}_{s' \sim
# P(\cdot | s,a)} [v_T(s')], \quad \forall s \in \mathcal{S}
# $$

# +
policy_opt = np.full_like(policy, "")

actions_list = list(Action)

for x, y in each_state:
    i_max = np.argmax(
        [
            r[i_1d(x, y)] + gamma * np.sum(get_trans_prob(x, y, a) * v_t)
            for a in actions_list
        ]
    )

    policy_opt[x, y] = actions_list[i_max].value

# Account for different coordinate systems between numpy and our board
print(np.flipud(policy_opt.T))
# -

# The learned policy is printed above and its representation corresponds
# one-to-one with the shape and orientation of the original board given in the
# problem statement.
#
# Lastly, we can calculate the value function for this policy by re-calculating
# our $P$ matrix and then using the analytical solution:

# +
p_opt = np.array(
    [get_trans_prob(x, y, policy_opt[x, y]) for x, y in each_state],
)

v_opt = np.linalg.inv(np.identity(width**2) - gamma * p_opt) @ r

value_text = []
for val, (x, y) in zip(v_opt, each_state):
    value_text.append(rf"v^{{\pi_T}}(s_{{ {x},{y} }}) &= {val:g} \\")

value_text = "\n".join(value_text)
# -

# \begingroup
# \allowdisplaybreaks
# \begin{align*}
# `{python} Markdown(value_text)`
# \end{align*}
# \endgroup
