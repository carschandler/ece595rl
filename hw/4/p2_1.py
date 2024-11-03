import itertools

import numpy as np
from IPython.display import Markdown
from p1_1 import gamma, reward_fn, state_space, trans_fn

policy = {1: "g", 2: "g", 3: "h"}

expected_trans_fn = np.reshape(
    [
        trans_fn.sel(current_state=s, next_state=sn, action=policy[s])
        for s, sn in itertools.product(state_space, state_space)
    ],
    (3, 3),
)

expected_reward_fn = np.array(
    [reward_fn.sel(state=s, action=policy[s]) for s in state_space]
)

value_fn = (
    np.linalg.inv(np.identity(len(state_space)) - gamma * expected_trans_fn)
    @ expected_reward_fn
)

value_latex = Markdown(
    r"\\".join(
        [rf"V^{{\pi}}({s}) = {v:.5f}" for s, v in zip(list("123"), value_fn)]
    )
)
