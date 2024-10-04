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


r = np.zeros(width * width, dtype=int)
r[i_1d(*np.argwhere(board == BoardSpace.LIGHTNING).squeeze())] = -1
r[i_1d(*np.argwhere(board == BoardSpace.TREASURE).squeeze())] = 1


gamma = 0.95
v = np.linalg.inv(np.identity(width**2) - gamma * p) @ r
value_text = []
for i, val in enumerate(v):
    y, x = np.unravel_index(i, [width, width])
    value_text.append(rf"v^{{\pi}}(s_{{ {x},{y} }}) &= {val:g} \\")

value_text = "\n".join(value_text)

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


max_error = norm(v_t - v, ord=np.inf)


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

print(np.flipud(policy_opt.T))


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


seed = 1298319824791827491284982176
rng = np.random.default_rng(seed)

int_to_action = np.vectorize(lambda i: actions_list[i])

policy_0 = int_to_action(rng.integers(low=0, high=4, size=(width, width)))

print(policy_0)


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


v_policy_opt = evaluate_policy(policy_opt)
value_text = []
for val, (x, y) in zip(v_policy_opt, each_state):
    value_text.append(rf"v^{{\pi_T}}(s_{{ {x},{y} }}) &= {val:g} \\")

value_text = "\n".join(value_text)
