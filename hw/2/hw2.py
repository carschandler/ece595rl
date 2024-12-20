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
# # latex-tinytex: false
# format:
#   pdf:
#     monofont: "CommitMono Nerd Font"
# jupyter: python3
# ---
#
# ## Problem 1
#
# ### 1.1
#
# Any matrix with only nonzero eigenvectors is necessarily invertible. So, we seek
# to prove that $I - \gamma P^{\pi}$ has only nonzero eigenvectors.
#
# Using the properties of eigenvalues, if $\lambda$ is an eigenvalue of $P^{\pi}$,
# then $1 - \gamma \lambda$ is an eigenvalue of $I - \gamma P^{\pi}$. So, if we
# can show that $\gamma \lambda \ne 1$ for all $\lambda$, then we can prove the
# matrix is invertible. Since we are in the discounted setting, we know that $0 <
# \gamma < 1$. So, if we can show that $\lambda \le 1$, then we will have
# achieved our goal.
#
# We know that $P^{\pi}$ is a row-stochastic matrix, which is to say that each
# element must be in the interval $[0, 1]$ and the sum of the elements in each row
# must be $1$. Given the definition of the eigenvalue and eigenvector:
#
# $$
# \mathbf{Ax} = \lambda \mathbf{x}
# $$
#
# We begin by analyzing the RHS of the equation.
#
# If we consider some $\lambda > 1$ to exist, then by definition:
#
# $$
# \lVert \lambda \mathbf{x} \rVert_{\infty} = | \lambda | \lVert \mathbf{x} \rVert_{\infty}
# $$
#
# and since $\lambda > 1$, then
#
# $$
# \lVert \lambda \mathbf{x} \rVert_{\infty} = | \lambda | \lVert \mathbf{x} \rVert_{\infty} > \lVert \mathbf{x} \rVert_{\infty}
# $$
#
# Now, looking at the LHS,
#
# $$
# \lVert P^{\pi} \mathbf{x} \rVert_{\infty} \le 
# \lVert \mathbf{x} \rVert_{\infty}
# $$
#
# This is true because each element in the vector resulting from $P^{\pi}
# \mathbf{x}$ will be some linear combination of the components of $\mathbf{x}$
# where the scalars are at least $0$ and at most $1$ and sum to $1$. So, the
# greatest magnitude possible from this operation occurs in the case where a row
# of $P^{\pi}$ has a $1$ that aligns with the maximum magnitude of $\mathbf{x}$
# and is zero everywhere else.
#
# Now, we have proven that the LHS is less than or equal to $\lVert \mathbf{x}
# \rVert_{\infty}$ and that the RHS is greater than $\lVert \mathbf{x}
# \rVert_{\infty}$. But if these two sides of the equation are to be equal, then
# this is impossible. Therefore, it is impossible that $\lambda > 1$ for a
# row-stochastic matrix.
#
# So if $\lambda \le 1$ for all eigenvalues of $P^{\pi}$, then $1 - \gamma \lambda >
# 0$, because $0 < \gamma < 1$. Therefore, all eigenvalues of $I - \gamma P^{\pi}$
# are nonzero, which means that $I - \gamma P^{\pi}$ must also be invertible.
#
# ### 1.2
#
# Beginning with the Bellman consistency equation:
#
# $$
# \begin{aligned}
# v^{\pi}(s) &= \mathbb{E}_{a \sim \pi(s)} \left[ R(s,a) + \gamma \mathbb{E}_{s'
# \sim P(\cdot | s,a)} \left[ v^{\pi}(s') \right] \right] \\
# &= \sum_{a \in \mathcal{A}} \left[ \mathbb{P}(a | s) \left( R(s,a) + \gamma \sum_{s' \in \mathcal{S}} \left[ \mathbb{P}(s' | a,s) v^{\pi}(s') \right] \right) \right] \\
# &= \sum_{a \in \mathcal{A}} \left[ \mathbb{P}(a | s)  R(s,a) \right] + \gamma \sum_{s' \in \mathcal{S}} \left[  v^{\pi}(s') \sum_{a \in \mathcal{A}} \left[ \mathbb{P}(a | s) \mathbb{P}(s' | a,s) \right] \right] \\
# \end{aligned}
# $$
#
# We can define some new variables under the policy $\pi$:
#
# $$
# R^{\pi}(s)  = \sum_{a \in \mathcal{A}} \left[ \mathbb{P}(a | s)  R(s,a) \right]
# $$
#
# $$
# P^{\pi}(s, s') = \sum_{a \in \mathcal{A}} \left[ \mathbb{P}(a | s) \mathbb{P}(s' | a,s) \right]
# $$
#
# These are just like before, except they are now a weighted average across all
# possible actions. We could define a function $\pi(a, s) = \mathbb{P}(a | s)$ and
# substitute this into the equations above as well.
#
# So we have
#
# $$
# v^{\pi}(s) = R^{\pi}(s) + \gamma \sum_{s' \in \mathcal{S}} \left[  P^{\pi}(s, s') v^{\pi}(s') \right]
# $$
#
# where $P^{\pi}(s, s') = \mathbb{P}(s' | s)$.
#
# We can form matrices to represent these across all states:
#
# $$
# \vec{v}^{\pi} = [v^\pi(s_1) \ ... \  v^{\pi}(s_n)]^T
# $$
#
# $$
# \vec{R}^{\pi} = [R^\pi(s_1) \ ... \  R^{\pi}(s_n)]^T
# $$
#
# $$
# P^{\pi} = \begin{bmatrix}
# P^\pi(s_1, s_1') & ... & P^{\pi}(s_1, s_n') \\
# \vdots & \ddots & \vdots \\
# P^{\pi}(s_n, s_1') & ... & P^{\pi}(s_n, s_n') \\
# \end{bmatrix}
# $$
#
# Which allows us to write the equation in matrix form as 
#
# $$
# \begin{aligned}
# \vec{v}^{\pi} &= \vec{R}^{\pi} + \gamma P^{\pi} \vec{v}^{\pi} \\
# \vec{v}^{\pi} - \gamma P^{\pi} \vec{v}^{\pi} &= \vec{R}^{\pi} \\
# (I - \gamma P^{\pi}) \vec{v}^{\pi} &= \vec{R}^{\pi} \\
# \vec{v}^{\pi}&= (I - \gamma P^{\pi})^{-1} \vec{R}^{\pi} \\
# \end{aligned}
# $$
#
# ### 1.3
#
# Since the reward is stochastic now, we need to start with the definition of the
# value function and derive a modified Bellman consistency equation:
#
# $$
# \begin{aligned}
# v^{\pi}(s) &= \mathbb{E}_{\pi, P, R}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) | s_0 = s \right] \\
# &= \mathbb{E}_{\pi, P, R} \left[ R(s_0, a_0) + \sum_{t=1}^{\infty} \gamma^t R(s_t, a_t) | s_0 = s \right] \\
# &= \sum_{a \sim \pi} \left[ \pi(a | s) \sum_{s', r} \left[ p(s', r | s, a) \left( r(s, a) + \gamma v^{\pi}(s') \right) \right] \right], \quad \forall s \in \mathcal{S} \\
# &= \sum_{a \sim \pi} \left[ \pi(a | s) \mathbb{E}_r \left[ r(s, a) + \mathbb{E}_{s'} \left[ \gamma v^{\pi}(s') \right] \right] \right], \quad \forall s \in \mathcal{S} \\
# \end{aligned}
# $$
#
# Our $R^{\pi}(s)$ is now defined as:
#
# $$
# R^{\pi}(s) = \sum_{a \in \mathcal{A}, r \in \mathcal{R}} \left[ \mathbb{P}(a | s) \mathbb{P}(r | a,s) r \right]
# $$
#
# Where $r$ is the value of the current reward in the summation. So, we sum over
# all possible actions as well as all possible rewards and take into account the
# probability of each action and reward as defined by the stochastic reward
# function $R$ and stochastic policy $\pi$ to give us an expectation across all
# possible action and reward combinations for a given state.
#
# From here, we can use the same logic from [1.2] to form our matrices, and the
# final result is still:
#
# $$
# \vec{v}^{\pi} = (I - \gamma P^{\pi})^{-1} \vec{R}^{\pi}
# $$
#
# ## Problem 2
#
# ### 2.1
#
# We wish to prove that
#
# $$
# \left| \max_{x \in X} g_1(x) - \max_{x \in X} g_2(x) \right|
# \le \max_{x \in X} \left| g_1(x) - g_2(x) \right|
# $$
#
# It is true that
#
# $$
# g_1(x) \le | g_1(x) - g_2(x) | + g_2(x)
# $$
#
# Since if $g_1(x) \ge g_2(x)$, equality holds above, and if $g_1(x) \le g_2(x)$,
# strict inequality holds. It therefore also holds that
#
# $$
# \max_{x \in X} g_1(x) \le \max_{x \in X}\left(| g_1(x) - g_2(x) | + g_2(x)\right)
# $$
#
# and by the triangle inequality (or at least the analagous inequality for the
# $\max$ operator)
#
# $$
# \max_{x \in X} g_1(x) \le \max_{x \in X}\left(| g_1(x) - g_2(x) | + g_2(x)\right) \le \max_{x \in X}(| g_1(x) - g_2(x) |) + \max_{x \in X} g_2(x)
# $$
#
# so then
#
# $$
# \max_{x \in X} g_1(x) - \max_{x \in X} g_2(x) \le \max_{x \in X}(| g_1(x) - g_2(x) |) 
# $$
#
# And if we repeat this process, switching the order of $g_1$ and $g_2$:
#
# $$
# \begin{gathered}
# g_2(x) \le | g_1(x) - g_2(x) | + g_1(x) \\
# ...\\
# \max_{x \in X} g_2(x) - \max_{x \in X} g_1(x) \le \max_{x \in X}(| g_1(x) - g_2(x) |)
# \end{gathered}
# $$
#
# And with the inequality proven for both orders of subtraction, we can combine
# the two conclusions by bringng in absolute value to the LHS:
#
# $$
# \left| \max_{x \in X} g_1(x) - \max_{x \in X} g_2(x) \right|
# \le \max_{x \in X} \left| g_1(x) - g_2(x) \right| \quad \blacksquare
# $$
#
# ### 2.2
#
# We wish to prove that
#
# $$
# \max_{x \in X, y \in Y} f(x, g(y)) \ge \max_{x \in X} f(x, \max_{y \in Y} g(y))
# $$
#
# for two scalar-valued functions $f : X \times Z \to \mathbb{R}$ and $g : Y \to Z
# \subseteq \mathbb{R}$.
#
# By definition,
#
# $$
# \max_{x \in X, y \in Y} f(x, g(y)) \ge f(x', g(y')), \quad \forall x' \in X, y' \in Y
# $$
#
# Define $y^*$ as:
#
# $$
# g(y^*) = \max_{y \in Y} g(y)
# $$
#
# So 
#
# $$
# \max_{x \in X} f(x, \max_{y \in Y} g(y))
# = \max_{x \in X} f(x, g(y^*))
# $$
#
# Now define $x^*$ as:
#
# $$
# f(x^*, y^*) = \max_{x \in X} f(x, g(y^*))
# $$
#
# But $x^* \in X$ and $y^* \in Y$ are just elements in their respective spaces,
# and we have already declared that
#
# $$
# \max_{x \in X, y \in Y} f(x, g(y)) \ge f(x', g(y')), \quad \forall x' \in X, y' \in Y
# $$
#
# for all elements in these spaces. So, if we just consider $x^*$ as some $x' \in
# X$ and the same for $y^*$, then:
#
# $$
# \max_{x \in X, y \in Y} f(x, g(y)) \ge f(x^*, g(y^*)) = \max_{x \in X} f(x, \max_{y \in Y} g(y)) \quad \blacksquare
# $$
#
# Intuitively, the operator on the LHS sweeps a 2D plane in its search for the
# maximum, while the RHS first sweeps a 1D line $Y$ and then sweeps a single line
# of the 2D plane in search for the maximum, so naturally there are more maxima
# that the LHS may find in comparison.
#
# ## Problem 3
#
# ### 3.1
#
# Given $v_{t=6}$, we can perform the following step in the value iteration 
#
# $$
# v_{7}(s) = \max_{a \in \mathcal{A}} \left[ R(s,a) + \gamma \mathbb{E}_{s' \sim
# P(\cdot | s,a)} [v_6(s')] \right], \quad \forall s \in \mathcal{S}
# $$
#
# We can implement this in Python:

# +
import numpy as np
from numpy.typing import NDArray

states = list("bc")
actions = list("xy")


def p(s, a):
    match (s, a):
        case ("b", "x"):
            return np.array([1.0, 0.0])
        case ("b", "y"):
            return np.array([0.2, 0.8])
        case ("c", "x"):
            return np.array([0.0, 1.0])
        case ("c", "y"):
            return np.array([0.6, 0.4])
        case _:
            raise Exception("Invalid state-action combination")


def reward(s, a):
    match (s, a):
        case ("b", "x"):
            return 0
        case ("b", "y"):
            return 0
        case ("c", "x"):
            return 1
        case ("c", "y"):
            return 1
        case _:
            raise Exception("Invalid state-action combination")


def opt_value(v_current: NDArray, n_iterations=1):
    v_t = v_current.copy()
    r_t = np.array([np.nan, np.nan])
    v_expected_t = np.array([np.nan, np.nan])
    optimal_actions = []

    for t in range(n_iterations):
        i_action = np.array(
            [
                np.argmax(
                    [reward(s, a) + np.sum(p(s, a) * v_t) for a in actions]
                )
                for s in states
            ]
        ).squeeze()

        r_t = np.array(
            [reward(s, actions[i_action[i]]) for i, s in enumerate(states)]
        )
        v_expected_t = np.array(
            [
                np.sum(p(s, actions[i_action[i]]) * v_t)
                for i, s in enumerate(states)
            ]
        )
        optimal_actions = [actions[i] for i in i_action]
        v_t = r_t + v_expected_t

    return v_t, r_t, v_expected_t, optimal_actions


v_opt_valiter_policy, r_opt, v_expected_opt, optimal_actions = opt_value(
    v_current=np.array([10, 5])
)
# -

# The value function at $t=7$ will be:
#
# $$
# v_7(b) = `{python} f"{v_expected_opt[0]:g}"` \gamma, \quad
# v_7(c) = 1 + `{python} f"{v_expected_opt[1]:g}"` \gamma
# $$
#
# ### 3.2
#
# Since we are given $v^{\pi_8}$, we have already performed the policy evaluation
# step, so we just need to improve the policy using:
#
# $$
# \pi_{9}(s) = \arg \max_{a \in \mathcal{A}} [R(s,a) + \gamma \mathbb{E}_{s' \sim
# P(\cdot | s,a)} [v^{\pi_8}(s')], \quad \forall s \in \mathcal{S}
# $$
#
# Our Python implementation already keeps track of the action corresponding to the
# optimal values in the `optimal_actions` output:

v_opt_valiter_policy, r_opt, v_expected_opt, optimal_actions = opt_value(
    v_current=np.array([5, 15])
)

# The improved policy at $t=9$ will be:
#
# $$
# \pi_9(b) = `{python} optimal_actions[0]`, \quad
# \pi_9(c) = `{python} optimal_actions[1]`
# $$
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
for i, val in enumerate(v_t):
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
# Therefore, we can use the same $T = `{python} n_iterations`$ as before.

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

v_valiter = v_t
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
            r[i_1d(x, y)]
            + gamma * np.sum(get_trans_prob(x, y, a) * v_valiter)
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
# ::: {.callout-note}
#
# #### Note
#
# The actions prescribed by this policy do not have any meaning for the special
# spaces where the user cannot move. The actions for these spaces are UP simply
# because that was the first action in the enumeration of actions.
#
# :::
#
# Lastly, we can calculate the value function for this policy by re-calculating
# our $P$ matrix and then using the analytical solution:

# +
p_opt = np.array(
    [get_trans_prob(x, y, policy_opt[x, y]) for x, y in each_state],
)

v_opt_valiter_policy = (
    np.linalg.inv(np.identity(width**2) - gamma * p_opt) @ r
)

value_text = []
for val, (x, y) in zip(v_opt_valiter_policy, each_state):
    value_text.append(rf"v^{{\pi_T}}(s_{{ {x},{y} }}) &= {val:g} \\")

value_text = "\n".join(value_text)
# -

# The $v^{\pi_T}(s)$ is shown for each state below:
#
# \begingroup
# \allowdisplaybreaks
# \begin{align*}
# `{python} Markdown(value_text)`
# \end{align*}
# \endgroup
#
#
# ## 4.3
#
# To begin our policy iteration algorithm, we initialize the policy with a uniform
# random distribution over $\mathcal{A}$:

# +
seed = 1298319824791827491284982176
rng = np.random.default_rng(seed)

int_to_action = np.vectorize(lambda i: actions_list[i])

policy_0 = int_to_action(rng.integers(low=0, high=4, size=(width, width)))

print(policy_0)


# -

# The policy is shown above using the original board shape and orientation.
#
# Now, for each iteration, we evaluate $\pi_t$ by computing $v^{\pi_t}$ and then
# we improve the policy by updating its action for each state so that it yields
# the maximum return according to:
#
# $$
# \pi_{t + 1}(s) = \arg \max_{a \in \mathcal{A}} [R(s,a) + \gamma \mathbb{E}_{s' \sim
# P(\cdot | s,a)} [v^{\pi_t}(s')], \quad \forall s \in \mathcal{S}
# $$
#
# To determine how many iterations are necessary, we use the following theorem:
#
# $$
# \lVert v^{\pi_T} - v^* \rVert_{\infty} \le \gamma^T \lVert v^{\pi_0} - v^*
# \rVert_{\infty}
# $$
#
# We can substitute our desired accuracy $\varepsilon$ in and solve for $T$:
#
# $$
# T \ge \frac{\log \left( \frac{\lVert v^{\pi_0} - v^*
# \rVert_{\infty}}{\varepsilon} \right)}{\log (\frac{1}{\gamma})}
# $$
#
# This is the same form we are used to seeing.
#
# Using the same logic as before, we can bound $v^*$ by:
#
# $$
# \Vert v^{*} \Vert_{\infty} \le 1 \cdot \sum_{t=0}^{\infty} \gamma^t = \frac{1}{1 - \gamma}
# $$
#
# Therefore:
#
# $$
# \lVert v^{\pi_0} - v^* \rVert_{\infty} \le \lVert v^{\pi_0} \rVert_{\infty} +
# \frac{1}{1-\gamma}
# $$
#
# So we can get a conservative upper bound on $T$ by substituting this into the
# inequality solved for $T$:
#
# $$
# T \ge \frac{\log \left( \frac{\lVert v^{\pi_0} \rVert_{\infty} +
# (1 - \gamma)^{-1}}{\varepsilon} \right)}{\log \left(\frac{1}{\gamma}\right)}
# $$
#
# Depending on how good our randomly generated initial policy is, $T$ may vary.
#
# We solve for $T$ and perform our iterations:

# +
def evaluate_policy(policy):
    p_policy = np.array(
        [get_trans_prob(x, y, policy[x, y]) for x, y in each_state],
    )

    v_policy = np.linalg.inv(np.identity(width**2) - gamma * p_policy) @ r

    return v_policy


def improve_policy(policy, v_policy):
    for x, y in each_state:
        i_max = np.argmax(
            [
                r[i_1d(x, y)]
                + gamma * np.sum(get_trans_prob(x, y, a) * v_policy)
                for a in actions_list
            ]
        )
        policy[x, y] = actions_list[i_max]


policy_t = policy_0

v_policy_0 = evaluate_policy(policy_0)

n_iterations = int(
    np.ceil(
        np.log((norm(v_policy_0, np.inf) + (1 / (1 - gamma))) / epsilon)
        / np.log(1 / gamma)
    )
)

for i in range(n_iterations):
    v_policy_t = evaluate_policy(policy_t)

    improve_policy(policy_t, v_policy_t)

policy_opt = policy_t

print(np.flipud(policy_opt.T))
# -

# The learned policy is shown above. The number of iterations was $T =
# `{python} n_iterations`$. We can evaluate the policy to view the value function:

# +
v_policy_opt = evaluate_policy(policy_opt)
value_text = []
for val, (x, y) in zip(v_policy_opt, each_state):
    value_text.append(rf"v^{{\pi_T}}(s_{{ {x},{y} }}) &= {val:g} \\")

value_text = "\n".join(value_text)
# -

# The value $v^{\pi_T}(s)$ is shown for each state below:
#
# \begingroup
# \allowdisplaybreaks
# \begin{align*}
# `{python} Markdown(value_text)`
# \end{align*}
# \endgroup
#
